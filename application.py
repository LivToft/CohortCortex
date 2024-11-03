from typing import Any
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import json
import ast
import anthropic
from PIL import Image, ImageTk

os.environ['ANTHROPIC_API_KEY'] = "YOUR_API_KEY"

class Patient:
    def __init__(self, patient_index: int, patient_data: pd.DataFrame):
        self.patient_index = patient_index
        self.patient_data = patient_data
        self.score = 0.0

class RuleBuilder:
    def __init__(self, rule_description: str):
        self.rule_description = rule_description

    def build_rule(self) -> dict[str, Any]:
        """
        Build a rule from a description.

        Returns:
            dict[str, Any]: The rule.
        """
        client = anthropic.Anthropic()
        system_prompt = """
            You are a top-of-the-line symbolic translator for clinical trials.
            You were built to solve the problem that, while patient data is well-defined
            and regularly structured, clinical trial definitions are not.

            You take natural-language clinical trial definitions that contain patient
            inclusion and exclusion criteria and convert them into a well-defined set of
            symbols that a program will be able to use to quickly parse well-formatted
            patient data files and assign a score to each trial based on the inclusion
            and exclusion criteria for the given trial.

            Your responses will always take the following format, without any extraneous
            prepended nor appended text:

            {
            "response": "rules",
            "inclusion_criterium": [
                {
                "rule": {
                        "type": "age",
                        "min": 58,
                        "max": 70
                    },
                    "weight": 1.0
                }
                },
                {
                ...
                }
            ]
            "exclusion_criterium": [
                {
                "rule": {
                        "type": "gender",
                        "gender": 1
                    }
                },
                {
                ...
                }
            ]
            }

            Specifications:
            - Every response must include a list of inclusion rules and exclusions rules
            - Both types of rules will have a list of criterium for the program to consider
            - The exclusion criterium will be considered to always be mandatory (1.0 weight)
            - The inclusion criterium can be more lenient, and you will assess how important the inclusion criteria is by assigning a weight from 0.0 to 1.0, which the program will consider when assigning a score to each potential patient.
            - Below you will see the strict dictionary you must stick to. These are the only criteria the engine knows how to verify at the current moment. However, if you feel like you cannot deliver a sufficiently good ruleset given the current dictionary, you may include the "other" rule type, described below.

            Symbol Dictionary:
            age: used to specify an age criteria for a patient.
            {
            “type”: “age”,
            “min”: <int> [years] (optional),
            “max”: <int> [years] (optional)
            }

            gender: used to specify a gender criteria for a patient.
            {
            “type”: “gender”,
            “gender”: <int> [0 for male, 1 for female, 2 for either] (2 is equivalent to not making this a rule)
            }

            medications: used to find whether a patient is prescribed certain medications.
            {
            “type”: “medications”,
            “medications”: <list[str]> [the lowercase names of specific medications] (these will be substring matched against the list of patient medications)
            }

            Example of medications:
            drug
            potassium chloride
            d5w
            sodium chloride
            ns
            furosemide
            insulin
            iso-osmotic dextrose
            5% dextrose
            sw
            magnesium sulfate
            morphine sulfate
            acetaminophen
            heparin
            calcium gluconate

            preexisting_conditions: used to find whether a patient has had prior diagnoses or conditions.
            {
            “type”: “preexisting_conditions”,
            “icd9_codes: <list[str]> [the ICD-9 codes describing the relevant conditions]
            }


            other: **Not to be used unless deemed strictly necessary.** Will inform the developers to prioritize work on adding capability to the engine to parse a rule of this format.
            {
            "type": "other"
            ... [whatever arguments you think are appropriate]
            }



            Notes:
            - Your goal is to be thorough and accurate. Clinical trials have been designed by medical professionals like yourself, and often certain criteria are only alluded to or implied. You should use the intuition from your training data to be as thorough and accurate as possible.
            - If for any reason you recieve a request that is not obviously a clinical trial definition, simply respond with:
            {
            "response": "error",
            "message": "Invalid clinical trial definition."
            }
        """
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.rule_description
                        }
                    ]
                }
            ]
        )

        rule = json.loads(message.content[0].text)
        return rule

class Engine:
    def __init__(self, rules: dict[str, Any], patient_csv_fp: str, log_folder: str = None):
        self.rules = rules
        self.patient_csv_fp = patient_csv_fp
        self.log_folder = log_folder
        if self.log_folder:
            os.makedirs(self.log_folder, exist_ok=True)
            self.exclusion_log_fp = os.path.join(self.log_folder, "exclusion_reasons.txt")
            self.scores_log_fp = os.path.join(self.log_folder, "patient_scores.csv")
            # Open the exclusion log file
            self.exclusion_log_file = open(self.exclusion_log_fp, 'w')
            # Prepare to save scores
            self.patient_scores = []
        else:
            self.exclusion_log_file = None
            self.patient_scores = []

    def sort_patients(self) -> list[Patient]:
        """
        The engine takes in a set of rules and a CSV file containing patient data.

        It converts the CSV into a pandas DataFrame.

        It then iterates through the patients in the DataFrame and applies the rules to each patient to obtain a score.

        Finally, it sorts the patients by score and returns the patients in a list.
        """
        # Load the patient data from the CSV file
        patient_data = pd.read_csv(self.patient_csv_fp)

        # Create a list to store the patients
        patients = []

        # Iterate through the patients
        for index, row in patient_data.iterrows():
            patient = Patient(index, row)
            patients.append(patient)

        # Apply the appropriate rules to each patient
        for patient in patients:
            score = 0.0

            # Apply the inclusion criteria
            for inclusion_criterium in self.rules["inclusion_criterium"]:
                rule = inclusion_criterium["rule"]
                weight = inclusion_criterium["weight"]

                if rule["type"] == "age":
                    score += self.age(patient, min=rule.get("min"), max=rule.get("max")) * weight
                elif rule["type"] == "gender":
                    score += self.gender(patient, gender=rule["gender"]) * weight
                elif rule["type"] == "medications":
                    score += self.medications(patient, medications=rule["medications"]) * weight
                elif rule["type"] == "preexisting_conditions":
                    score += self.preexisting_conditions(patient, study_icd9_codes=rule["icd9_codes"]) * weight

            patient.score = score
            # Apply the exclusion criteria (set score to 0 if any exclusion criteria are met)
            for exclusion_criterium in self.rules["exclusion_criterium"]:
                rule = exclusion_criterium["rule"]

                if rule["type"] == "age":
                    if self.age(patient, min=rule.get("min"), max=rule.get("max")) == 0.0:
                        patient.score = 0.0
                        reason = f"Patient {patient.patient_index} excluded due to age ({patient.patient_data['age']}, needed between {rule.get('min')} and {rule.get('max')})"
                        if self.exclusion_log_file:
                            self.exclusion_log_file.write(reason + '\n')
                        break
                elif rule["type"] == "gender":
                    if self.gender(patient, gender=rule["gender"]) == 0.0:
                        patient.score = 0.0
                        reason = f"Patient {patient.patient_index} excluded due to gender ({patient.patient_data['gender']}, needed to be {'M' if rule['gender'] == 0 else 'F'})"
                        if self.exclusion_log_file:
                            self.exclusion_log_file.write(reason + '\n')
                        break
                elif rule["type"] == "medications":
                    if self.medications(patient, medications=rule["medications"]) > 0.0:
                        patient.score = 0.0
                        reason = f"Patient {patient.patient_index} excluded due to medications ({patient.patient_data['prescriptions']}, but couldn't have {rule['medications']})"
                        if self.exclusion_log_file:
                            self.exclusion_log_file.write(reason + '\n')
                        break
                elif rule["type"] == "preexisting_conditions":
                    if self.preexisting_conditions(patient, study_icd9_codes=rule["icd9_codes"]) > 0.0:
                        patient.score = 0.0
                        reason = f"Patient {patient.patient_index} excluded due to preexisting conditions ({patient.patient_data['icd9_codes']}, but couldn't have {rule['icd9_codes']})"
                        if self.exclusion_log_file:
                            self.exclusion_log_file.write(reason + '\n')
                        break
            # Save the patient's score
            self.patient_scores.append({
                'patient_index': patient.patient_index,
                'subject_id': patient.patient_data['subject_id'],
                'first_name': patient.patient_data['first_name'],
                'last_name': patient.patient_data['last_name'],
                'score': patient.score
            })

        # Sort the patients by score
        patients.sort(key=lambda x: x.score, reverse=True)
        
        # Save the patient scores to CSV
        if self.log_folder:
            scores_df = pd.DataFrame(self.patient_scores)
            scores_df.to_csv(self.scores_log_fp, index=False)
        
        # Close the exclusion log file
        if self.exclusion_log_file:
            self.exclusion_log_file.close()
        
        return patients

    def age(self, patient: Patient, *,
            min: int | None = None, max: int | None = None) -> float:
        """
        Used to specify an age criteria for a patient.

        Args:
            patient (Patient): The patient to evaluate.
            min (int): The minimum age (optional).
            max (int): The maximum age (optional).

        Returns:
            float: 1.0 if the patient's age is within the specified range, 0.0 otherwise.
        """
        age = patient.patient_data['age']
        if min is not None and age < min:
            return 0.0
        if max is not None and age > max:
            return 0.0
        return 1.0

    def gender(self, patient: Patient, gender: int) -> float:
        """
        Used to specify a gender criteria for a patient.

        Args:
            patient_data (Patient): The patient to evaluate
            gender (int): 0 for male, 1 for female, 2 for either

        Returns:
            float: 1.0 if the patient is of the specified gender, 0.0 otherwise.
        """
        if gender == 0:
            if patient.patient_data['gender'] == 'M':
                return 1.0
            else:
                return 0.0
        elif gender == 1:
            if patient.patient_data['gender'] == 'F':
                return 1.0
            else:
                return 0.0
        elif gender == 2:
            return 1.0
        else:
            return 0.0

    def medications(self, patient: Patient, medications: list[str] = None) -> float:
        """
        Used to find whether a patient is prescribed certain medications.

        Args:
            patient_data (Patient): The patient to evaluate
            medications (list[str]): List of medications

        Returns:
            float: between 1.0 and 0.0 for the percent the patient's medications that match the listed medications
        """
        # Columns to check in patient data
        columns = ['prescriptions', 'prescriptions_poe', 'prescriptions_generic']
        matches = 0

        for med in medications:
            # convert medication name to all lowercase
            med = med.lower().strip()

            for col in columns:
                for patient_med in patient.patient_data[col]:
                    if med in patient_med.lower().strip():
                        matches += 1

        total = len(medications)

        return matches / total if total > 0 else 0.0

    def preexisting_conditions(self, patient: Patient, study_icd9_codes: list[str]) -> float:
        """
        Used to find whether a patient has had prior diagnoses or conditions.

        Args:
            patient_data (Patient): The patient to evaluate
            study_icd9_codes (list[str]): List of ICD-9 codes

        Returns:
            float: between 1.0 and 0.0 for the percent the patient's prior conditions that match the listed ICD-9 codes
        """

        patient_icd9_codes = patient.patient_data['icd9_codes']

        # Standardize ICD-9 codes for clean string matching
        patient_icd9_codes = patient_icd9_codes.strip().lower()
        study_icd9_codes = [x.strip().lower() for x in study_icd9_codes]

        # Convert ICD-9 diagnostic codes to sets
        patient_icd9_codes = set(ast.literal_eval(patient_icd9_codes))
        study_icd9_codes = set(study_icd9_codes)

        # Count how many ICD-9 codes the patient matches to the study
        set_diff = patient_icd9_codes.intersection(study_icd9_codes)

        # Return a binary score of 1 or 0 if patient matches all or no criteria
        # Return a raw score for ranking
        prior_condition_score = len(set_diff) / len(study_icd9_codes) if len(study_icd9_codes) > 0 else 0.0

        return prior_condition_score

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Clinical Trial Patient Selector")
        self.geometry("800x600")
        self.patient_csv_fp = None
        self.rules = None
        self.log_folder = "logs"
        os.makedirs(self.log_folder, exist_ok=True)
        self.create_widgets()
        
    def create_widgets(self):
        header_font = ctk.CTkFont(family="Montserrat", size=20, weight="bold")
        body_font = ctk.CTkFont(family="Barlow", size=14)
        # Create and place the widgets
        # Logo
        logo_img = Image.open("cortex_logo.png").resize((200, 200))
        logo = ImageTk.PhotoImage(logo_img)
        self.logo_label = ctk.CTkLabel(self, image=logo, text="")
        self.logo_label.image = logo  # Keep a reference to prevent garbage collection
        self.logo_label.pack(pady=10)  # Add padding for alignment

        # Button to select patient CSV file
        self.select_csv_button = ctk.CTkButton(self, text="Select Patient CSV File", command=self.select_csv_file, fg_color="#0B3948", font=body_font)
        self.select_csv_button.pack(pady=10)
        
        # Label to display selected file path
        self.csv_file_label = ctk.CTkLabel(self, text="No file selected", font=body_font)
        self.csv_file_label.pack()
        
        # Text widget for rule description
        self.rule_label = ctk.CTkLabel(self, text="Enter Rule Description:", font=header_font)
        self.rule_label.pack(pady=10)
        
        self.rule_text = ctk.CTkTextbox(self, height=200, width=600)
        self.rule_text.pack()
        
        # Button to run the application
        self.run_button = ctk.CTkButton(self, text="Run", command=self.run_application, fg_color="#0B3948", font=body_font)
        self.run_button.pack(pady=10)
        
        # Text widget to display the output
        self.output_label = ctk.CTkLabel(self, text="Patients:", font=header_font)
        self.output_label.pack(pady=10)
        
        self.output_text = ctk.CTkTextbox(self, height=200, width=600)
        self.output_text.pack()
        
    def select_csv_file(self):
        # Open file dialog to select CSV file
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.patient_csv_fp = file_path
            self.csv_file_label.configure(text=os.path.basename(file_path))
        else:
            self.patient_csv_fp = None
            self.csv_file_label.configure(text="No file selected")
    
    def run_application(self):
        # Get the rule description from the text widget
        rule_description = self.rule_text.get("1.0", tk.END).strip()
        if not rule_description:
            messagebox.showerror("Error", "Please enter a rule description.")
            return
        if not self.patient_csv_fp:
            messagebox.showerror("Error", "Please select a patient CSV file.")
            return
        try:
            # Build the rules
            rule_builder = RuleBuilder(rule_description)
            rules = rule_builder.build_rule()
            
            # Save the parsed rules to the log folder
            rules_log_fp = os.path.join(self.log_folder, "parsed_rules.json")
            with open(rules_log_fp, 'w') as f:
                json.dump(rules, f, indent=4)
            
            # Create an instance of the engine
            engine = Engine(rules, self.patient_csv_fp, log_folder=self.log_folder)
            
            # Sort the patients
            patients = engine.sort_patients()
            
            # Display the patients in the output text widget
            self.output_text.delete("1.0", tk.END)
            num_patients = len(patients)
            num_included_patients = sum(patient.score > 0 for patient in patients)
            percentage_included = num_included_patients / num_patients * 100 if num_patients > 0 else 0
            
            output_lines = []
            for patient in patients:
                if patient.score > 0:
                    output_line = f"{patient.patient_data['first_name']} {patient.patient_data['last_name']}, ID: {patient.patient_data['subject_id']}. Score: {patient.score}"
                    output_lines.append(output_line)
            output_text = "\n".join(output_lines)
            self.output_text.insert(tk.END, output_text)
            
            # Save the scores and excluded reasons to logs
            # Assuming that the Engine class saves the logs
            self.show_popup("Success", f"Processed patients. Percentage included: {percentage_included:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_popup(self, title, message):
        body_font = ctk.CTkFont(family="Barlow", size=14)
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        popup.geometry("400x200")
        popup.configure(bg="0B3948")
        label = ctk.CTkLabel(popup, text=message, font=body_font, bg_color="#0B3948", wraplength=350)
        label.pack(pady=20)
        ok_button = ctk.CTkButton(popup, text="OK", command=popup.destroy, fg_color="#D81E5B")
        ok_button.pack(pady=10)

if __name__ == "__main__":
    app = Application()
    app.mainloop()

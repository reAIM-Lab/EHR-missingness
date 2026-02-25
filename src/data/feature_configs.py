codes_to_keep = {
    'Body Weight': ['LOINC/29463-7'],  
    'Heart Rate': ['LOINC/8867-4', 'SNOMED/364075005'],
    'Systolic blood pressure': ['LOINC/8480-6', 'SNOMED/271649006'],
    'Diastolic blood pressure': ['LOINC/8462-4', 'SNOMED/271650006'],
    'Respiratory rate': ['LOINC/9279-1'],
    'Oxygen saturation': ['LOINC/2708-6'],
    'Hemoglobin': ['LOINC/718-7', 'SNOMED/271026005', 'SNOMED/441689006'],
    'Hematocrit': ['LOINC/4544-3', 'LOINC/20570-8', 'LOINC/48703-3', 'SNOMED/28317006'],
    'Erythrocytes': ['LOINC/789-8', 'LOINC/26453-1'],
    'Leukocytes': ['LOINC/20584-9', 'LOINC/6690-2'],
    'Platelets': ['LOINC/26515-7'],
    'Sodium': ['LOINC/2951-2', 'LOINC/2947-0', 'SNOMED/25197003'],
    'Potassium': ['LOINC/2823-3', 'SNOMED/312468003', 'LOINC/6298-4', 'SNOMED/59573005'],
    'Chloride': ['LOINC/2075-0', 'SNOMED/104589004', 'LOINC/2069-3'],
    'Carbon dioxide, total': ['LOINC/2028-9'],
    'Calcium': ['LOINC/17861-6', 'SNOMED/271240001'],
    'Glucose': ['LOINC/2345-7', 'SNOMED/166900001', 'LOINC/2339-0', 'SNOMED/33747003', 'LOINC/14749-6'],
    'Urea nitrogen': ['LOINC/3094-0', 'SNOMED/105011006'],
    'Creatinine': ['LOINC/2160-0', 'SNOMED/113075003'],
    'pO2': ['LOINC/2703-7'],
    'pCO2': ['LOINC/2019-8'],
    'pH': ['LOINC/2744-1'],
    'Bicarbonate': ['LOINC/1960-4'],
    'Magnesium': ['LOINC/19123-9'],
    'Phosphate': ['LOINC/2777-1'],
    'Bilirubin': ['LOINC/1975-2'],
    'Albumin': ['LOINC/1751-7'],
    'INR': ['LOINC/6301-6'],
    'aPTT': ['LOINC/14979-9'],
    'Monocytes': ['LOINC/5905-5'],
    'Lymphocytes': ['LOINC/26478-8'],
    'Neutrophils': ['LOINC/26511-6'],
    'Lactate': ['LOINC/32693-4', 'LOINC/2518-9'],
    'Troponin I': ['LOINC/10839-9'], 
    'Troponin T': ['LOINC/6598-7'],
}

codes_to_keep_mimic = {
    'Platelets': ['LOINC/777-3'],
    'Hemoglobin': ['LOINC/718-7', 'SNOMED/271026005', 'SNOMED/441689006'],
    'Hematocrit': ['LOINC/4544-3', 'LOINC/20570-8', 'LOINC/48703-3', 'SNOMED/28317006'],
    'MCH': ['LOINC/785-6'],
    'MCV': ['LOINC/787-2'],
    'Erythrocytes': ['LOINC/789-8', 'LOINC/26453-1'],
    'Leukocytes': ['LOINC/20584-9', 'LOINC/6690-2'],
    'Sodium': ['LOINC/2951-2', 'LOINC/2947-0', 'SNOMED/25197003'],
    'Potassium': ['LOINC/2823-3', 'SNOMED/312468003', 'LOINC/6298-4', 'SNOMED/59573005'],
    'Chloride': ['LOINC/2075-0', 'SNOMED/104589004', 'LOINC/2069-3'],
    'Calcium': ['LOINC/17861-6', 'SNOMED/271240001'],
    'Glucose': ['LOINC/2345-7', 'SNOMED/166900001', 'LOINC/2339-0', 'SNOMED/33747003', 'LOINC/14749-6'],
    'Urea nitrogen': ['LOINC/3094-0', 'SNOMED/105011006'],
    'Creatinine': ['LOINC/2160-0', 'SNOMED/113075003'],
    'Bicarbonate': ['LOINC/1963-8'],
    'Magnesium': ['LOINC/19123-9'],
    'Phosphate': ['LOINC/2777-1'],
    'Lactate': ['LOINC/32693-4', 'LOINC/2518-9'],
    'PaO2': ['LOINC/11556-8'],
    'PaCO2': ['LOINC/11557-6'],
    'pH': ['LOINC/11558-4'],
    'Monocytes': ['LOINC/5905-5'],
    'Neutrophils': ['LOINC/770-8'],
    'Lymphocytes': ['LOINC/736-9'],
    'Albumin': ['LOINC/1751-7'],
    'Bilirubin': ['LOINC/1975-2'],
    'INR': ['LOINC/6301-6'],
    'aPPT': ['LOINC/14979-9'],
    'Troponin T': ['LOINC/6598-7'],
}

# TODO: missingness rate ablation with feature groups
# Add ablation without prompting
# Use original data distribution

required_categories = [
    'Body Weight',
    'Heart Rate', 
    'Respiratory rate', 
    'Systolic blood pressure', 
    'Diastolic blood pressure', 
    'Oxygen saturation',
    'Hemoglobin',
    'Hematocrit',
    'Platelets',
    'Leukocytes',
    'Erythrocytes',
    ]

required_categories_mimic = [
    'Platelets',
    'Hemoglobin',
    'Hematocrit',
    'MCH',
    'MCV',
    'Leukocytes',
    'Erythrocytes',
]

clip_range = {
    'Body Weight': [350, 10000],
    'Heart Rate': [5, 300],
    'Systolic blood pressure': [20, 300],
    'Diastolic blood pressure': [20, 300],
    'Respiratory rate': [1, 100],
    'Oxygen saturation': [70, 100],
    'Hemoglobin': [1, 20],
    'Hematocrit': [10, 70],
    'Erythrocytes': [1, 12],  # LOINC/789-8, LOINC/26453-1
    'Leukocytes': [0.1, 150],  # LOINC/20584-9, LOINC/6690-2
    'Platelets': [1, 1000],  # LOINC/777-3, SNOMED/61928009
    'Sodium': [100, 200],  # LOINC/2951-2, LOINC/2947-0, SNOMED/25197003
    'Potassium': [0.1, 10],  # LOINC/2823-3, SNOMED/312468003, LOINC/6298-4, SNOMED/59573005
    'Chloride': [50, 200],  # LOINC/2075-0, SNOMED/104589004, LOINC/2069-3
    'Carbon dioxide, total': [10, 100],  # LOINC/2028-9 (mmol/L), 23-28 (Integer)
    'Calcium': [1, 20],  # LOINC/17861-6, SNOMED/271240001
    'Glucose': [10, 1000],  # LOINC/2345-7, SNOMED/166900001, LOINC/2339-0, SNOMED/33747003, LOINC/14749-6
    'Urea nitrogen': [1, 200],  # LOINC/3094-0, SNOMED/105011006
    'Creatinine': [0.1, 10],  # LOINC/2160-0, SNOMED/113075003
    'pO2': [20, 600],
    'pCO2': [10, 120],
    'Bicarbonate': [5, 50],
    'Lactate': [0.2, 20],
    'Magnesium': [0.5, 5.0],
    'Phosphate': [0.5, 10],
    'Troponin T': [0, 100],
    'PaO2': [0, 500],
    'PaCO2': [0, 150],
    'pH': [6.8, 7.8],
    'Monocytes': [0, 30],
    'Neutrophils': [0, 100],
    'Lymphocytes': [0, 100],
    'Albumin': [0.5, 10],
    'Bilirubin': [0, 50],
    'INR': [0.5, 20],
    'aPTT': [5, 200],
}

codes_to_keep = {
    'Body Weight': ['LOINC/29463-7'],  
    'Heart Rate': ['LOINC/8867-4', 'SNOMED/364075005'],
    'Systolic blood pressure': ['LOINC/8480-6', 'SNOMED/271649006'],
    'Diastolic blood pressure': ['LOINC/8462-4', 'SNOMED/271650006'],
    'Respiratory rate': ['LOINC/9279-1'],
    'Oxygen saturation': ['LOINC/LP21258-6', 'LOINC/20564-1', 'LOINC/59408-5', 'LOINC/2708-6'],
    'Hemoglobin': ['LOINC/718-7', 'SNOMED/271026005', 'SNOMED/441689006'],
    'Hematocrit': ['LOINC/4544-3', 'LOINC/20570-8', 'LOINC/48703-3', 'SNOMED/28317006'],
    'Erythrocytes': ['LOINC/789-8', 'LOINC/26453-1'],
    'Leukocytes': ['LOINC/20584-9', 'LOINC/6690-2'],
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
    'Bicarbonate': ['LOINC/1960-4'],
    'Magnesium': ['LOINC/19123-9'],
    'Phosphate': ['LOINC/2777-1'],
}

required_categories = [
    'Heart Rate', 
    'Respiratory rate', 
    'Systolic blood pressure', 
    'Diastolic blood pressure', 
    'Oxygen saturation', 
    'Body Weight'
    ]

clip_range = {
    'Body Weight': [350, 10000],
    #'Body Height': [5, 100],
    #'BMI': [10, 100],
    #'Body Surface Area': [0.1, 100],
    'Heart Rate': [5, 300],
    'Systolic blood pressure': [20, 300],
    'Diastolic blood pressure': [20, 300],
    #'Body temperature': [80, 120],
    'Respiratory rate': [1, 100],
    'Oxygen saturation': [1, 100],
    'Hemoglobin': [1, 20],
    'Hematocrit': [10, 100],
    'Erythrocytes': [1, 10],  # LOINC/789-8, LOINC/26453-1
    'Leukocytes': [1, 100],  # LOINC/20584-9, LOINC/6690-2
    'Platelets': [10, 1000],  # LOINC/777-3, SNOMED/61928009
    'Sodium': [100, 200],  # LOINC/2951-2, LOINC/2947-0, SNOMED/25197003
    'Potassium': [0.1, 10],  # LOINC/2823-3, SNOMED/312468003, LOINC/6298-4, SNOMED/59573005
    'Chloride': [50, 200],  # LOINC/2075-0, SNOMED/104589004, LOINC/2069-3
    'Carbon dioxide, total': [10, 100],  # LOINC/2028-9 (mmol/L), 23-28 (Integer)
    'Calcium': [1, 20],  # LOINC/17861-6, SNOMED/271240001
    'Glucose': [10, 1000],  # LOINC/2345-7, SNOMED/166900001, LOINC/2339-0, SNOMED/33747003, LOINC/14749-6
    'Urea nitrogen': [1, 200],  # LOINC/3094-0, SNOMED/105011006
    'Creatinine': [0.1, 10],  # LOINC/2160-0, SNOMED/113075003
    #'pH': [6.8, 8.0],
    'pO2': [20, 600],
    'pCO2': [10, 120],
    'Bicarbonate': [5, 50],
    #'Lactate': [0.2, 20],
    'Magnesium': [0.5, 5.0],
    'Phosphate': [0.5, 10],
    #'Troponin': [0, 100],
}

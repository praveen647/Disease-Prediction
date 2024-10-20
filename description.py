def get_disease_description(disease_name):
    dismap = {'Fungal infection' : "Description: Infections caused by fungi, affecting skin, nails, and mucous membranes. Prescription: Antifungal medications like Fluconazole or Clotrimazole.",
          'Allergy' : "Description: An immune response to allergens such as pollen, dust, or certain foods. Prescription: Antihistamines (e.g., Cetirizine), corticosteroids, and decongestants.",
          'GERD' : "Description: A chronic condition where stomach acid flows back into the esophagus. Prescription: Proton pump inhibitors (PPIs) like Omeprazole or H2 blockers like Ranitidine.", 
          'Chronic cholestasis' : "Description: A condition characterized by impaired bile flow. Prescription: Ursodeoxycholic acid to improve bile flow and reduce symptoms.",
          'Drug Reaction' : "Description: Adverse reactions to medications, ranging from mild to severe. Prescription: Antihistamines for mild reactions; corticosteroids for severe cases.",
          'Peptic ulcer diseae' : "Description: Sores that develop on the lining of the stomach or the upper part of the small intestine. Prescription: PPIs like Omeprazole and antibiotics for H. pylori eradication.",
          'AIDS' : "Description: A chronic condition caused by the HIV virus that weakens the immune system. Prescription: Antiretroviral therapy (ART), including medications like Tenofovir and Efavirenz.",
          'Diabetes ' : "Description: A chronic condition that affects how the body processes blood sugar (glucose). Prescription: Insulin or oral medications like Metformin.",
          'Gastroenteritis' : "Description: Inflammation of the stomach and intestines, often caused by infections. Prescription: Hydration, and in some cases, antibiotics (if bacterial).",
          'Bronchial Asthma' : "Description: A condition causing wheezing, shortness of breath, and coughing due to airway inflammation. Prescription: Inhaled corticosteroids (e.g., Fluticasone) and bronchodilators (e.g., Albuterol).",
          'Hypertension ' : "Description: High blood pressure, increasing the risk of heart disease and stroke. Prescription: Antihypertensive medications like Lisinopril or Amlodipine.", 
          'Migraine' : "Description: Severe, recurring headaches often accompanied by nausea and sensitivity to light. Prescription: Triptans (e.g., Sumatriptan) or preventive medications like Propranolol.",
          'Cervical spondylosis' : "Description: Age-related wear and tear affecting spinal discs in the neck. Prescription: Pain relievers (e.g., NSAIDs) and physical therapy.",
          'Paralysis (brain hemorrhage)' : "Description: Loss of muscle function due to bleeding in the brain. Prescription: Depends on severity; may require rehabilitation and medications to control blood pressure.",
          'Jaundice' : "Description: Yellowing of the skin and eyes due to liver dysfunction or bile duct obstruction. Prescription: Treatment targets the underlying cause; may involve medications for liver health.",
          'Malaria' : "Description: A disease caused by parasites transmitted through mosquito bites. Prescription: Antimalarial medications like Chloroquine or Artemisinin-based combination therapies (ACTs).",
          'Chicken pox' : "Description: A highly contagious viral infection causing an itchy rash and flu-like symptoms. Prescription: Antihistamines for itching and Acetaminophen for fever; antiviral medications in severe cases.",
          'Dengue' : "Description: A mosquito-borne viral infection causing flu-like symptoms and severe headaches. Prescription: Supportive care; no specific antiviral treatment; hydration and pain relief with Acetaminophen.",
          'Typhoid' : "Description: A bacterial infection caused by Salmonella typhi, leading to fever and gastrointestinal issues. Prescription: Antibiotics such as Ciprofloxacin or Azithromycin.",
          'hepatitis A' : "Description: A viral infection affecting the liver, usually spread through contaminated food or water. Prescription: Supportive care; vaccination for prevention.",
          'Hepatitis B':"Description: A serious liver infection caused by the hepatitis B virus, often spread through bodily fluids. Prescription: Antivirals like Tenofovir or Entecavir.",
          'Hepatitis C':"Description: A viral infection affecting the liver, leading to chronic liver disease. Prescription: Direct-acting antiviral (DAA) medications like Sofosbuvir.", 
    'Hepatitis D':"Description: A liver infection caused by the hepatitis D virus, only occurring in those infected with hepatitis B. Prescription: Antiviral therapy targeting hepatitis B.", 
    'Hepatitis E':"Description: A viral infection caused by the hepatitis E virus, usually spread through contaminated water. Prescription: Supportive care; vaccination available in some countries.",
    'Alcoholic hepatitis':"Description: Inflammation of the liver caused by excessive alcohol consumption. Prescription: Corticosteroids or pentoxifylline in severe cases.",
    'Tuberculosis':"Description: A bacterial infection primarily affecting the lungs but can spread to other parts of the body. Prescription: A combination of antibiotics like Isoniazid, Rifampicin, Ethambutol, and Pyrazinamide.",
      'Common Cold':"Description: A viral infection of the upper respiratory tract. Prescription: Symptomatic relief with antihistamines and decongestants.", 
      'Pneumonia':"Description: Infection that inflames air sacs in one or both lungs, which may fill with fluid. Prescription: Antibiotics if bacterial; supportive care for viral pneumonia.",
    'Dimorphic hemmorhoids(piles)':"Description: Swollen veins in the lower rectum or anus, causing discomfort. Prescription: Topical treatments, fiber supplements, and sitz baths; surgical options in severe cases.",
     'Heart attack':"Description: A serious condition where blood flow to the heart is blocked. Prescription: Aspirin, beta-blockers, statins, and emergency interventions like angioplasty.",
       'Varicose veins':"Description: Swollen, twisted veins visible just under the surface of the skin. Prescription: Compression stockings, lifestyle changes, and sclerotherapy.",
        'Hypothyroidism':"Description: A condition where the thyroid gland does not produce enough hormones. Prescription: Thyroid hormone replacement therapy (e.g., Levothyroxine).",
     'Hyperthyroidism':"Description: A condition where the thyroid gland is overactive and produces too much hormone. Prescription: Antithyroid medications (e.g., Methimazole) or radioactive iodine.",
       'Hypoglycemia':"Description: Low blood sugar levels, leading to symptoms like dizziness and sweating. Prescription: Glucose tablets or gel, and dietary management to maintain stable blood sugar.",
    'Osteoarthristis':"Description: Degenerative joint disease causing pain and stiffness. Prescription: Pain relievers (e.g., NSAIDs) and physical therapy.", 
    'Arthritis':"Description: Inflammation of the joints, causing pain and stiffness. Prescription: NSAIDs, corticosteroids, and disease-modifying antirheumatic drugs (DMARDs) for rheumatoid arthritis.",
    '(vertigo) Paroymsal  Positional Vertigo':"Description: A common cause of vertigo, characterized by brief episodes of dizziness. Prescription: Vestibular rehabilitation therapy and medications like Meclizine.",
     'Acne':"Description: A skin condition that occurs when hair follicles become clogged with oil and dead skin cells. Prescription: Topical retinoids, benzoyl peroxide, and antibiotics.",
    'Urinary tract infection':"Description: An infection in any part of the urinary system, often causing painful urination. Prescription: Antibiotics like Nitrofurantoin or Trimethoprim-sulfamethoxazole.",
     'Psoriasis':"Description: A chronic autoimmune condition that causes rapid skin cell growth, leading to scaling. Prescription: Topical treatments (e.g., corticosteroids), phototherapy, and systemic medications.", 
     'Impetigo':"Description: A highly contagious bacterial skin infection characterized by red sores. Prescription: Antibiotic ointments (e.g., Mupirocin) or oral antibiotics in severe cases."
         }
    
    return dismap.get(disease_name)

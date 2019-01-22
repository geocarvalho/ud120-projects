 #!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
poi_people = []
has_email = []
has_salary = []
not_pay = []
poi_not_pay = []
for key, value in enron_data.iteritems():
    if enron_data[key]['poi']==1:
        poi_people.append(key)
        if enron_data[key]['total_payments']=='NaN':
            poi_not_pay.append(key)
    if enron_data[key]['salary']!='NaN':
        has_salary.append(key)
    if enron_data[key]['email_address']!='NaN':
        has_email.append(key)
    if enron_data[key]['total_payments']=='NaN':
        not_pay.append(key)
# print(enron_data['PRENTICE JAMES']['total_stock_value'])
# print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
# print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])
# print(enron_data['LAY KENNETH L']['total_payments'])
# print(len(has_salary), len(has_email))
# print(len(not_pay)/float(len(enron_data)))
# print(len(poi_not_pay)/float(len(poi_people)))
# print(len(enron_data)+10, len(not_pay)+10)
print(len(poi_people)+10, len(poi_not_pay)+10)
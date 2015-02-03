#!/usr/bin/python

from utils.data_loader import enron_data

num_people = len(enron_data)
person_keys = enron_data.keys()
person_values = enron_data.values()

poi_data = [person for person in person_values if person['poi'] is True]



def get_attribute_frequency(attribute, dict_list):
    res = 0
    for i in range(len(dict_list)):
        if dict_list[i][attribute] != 'NaN':
            res += 1
    return res



print "Number of people in the Enron dataset: {}".format(num_people)



person_1_data = person_values[0]
print "Data for {}: {}".format(person_keys[0], person_1_data)

num_features_per_person = len(person_1_data)
print "Number of features per person: {}".format(num_features_per_person)

print "# Persons of interest in dataset: {}".format(len(poi_data))

num_pois_in_text_list = 0
with open("data/poi_names.txt", "r") as f:
    lines = [line.rstrip() for line in f]    # All lines including the blank ones
    lines = [line for line in lines if line] # Non-blank lines

    num_pois_in_text_list = len(lines) - 1  # discount first line that consists of a link to the source for the names

print "# Persons of interset listed in text file: {}".format(num_pois_in_text_list)

james_prentice_stock_value = enron_data['PRENTICE JAMES']['total_stock_value']
print "James Prentice Stock value: {}".format(james_prentice_stock_value)

wesley_colwell_emails_to_pois = enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print "Emails sent by Wesley Colwell to POIs: {}".format(wesley_colwell_emails_to_pois)

jeff_skilling_stock_options_val = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print "Value of stock options exercised by Jeff Skilling: {}".format(
    jeff_skilling_stock_options_val
)


num_persons_with_salaries = get_attribute_frequency('salary', person_values)
num_persons_with_email = get_attribute_frequency('email_address', person_values)

print "Num persons with salaries: {}".format(num_persons_with_salaries)
print "Num persons with known email addresses: {}".format(num_persons_with_email)


pecentage_without_payments = (float)(
    num_people - get_attribute_frequency('total_payments', person_values)) / num_people

print "Percentage of individuals without recorded 'total_payments'" + \
       "data: {}".format(
           pecentage_without_payments
       )

percentage_of_pois_without_tot_payments_data = (float)(
    len(poi_data) - get_attribute_frequency('total_payments', poi_data)) / len(poi_data)


print "Percentage of POIs without recorded 'total payments'" + \
      "data: {}".format(percentage_of_pois_without_tot_payments_data)

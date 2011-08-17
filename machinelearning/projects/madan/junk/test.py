'''
Created on Nov 5, 2010

@author: Madan Thangavelu
'''
import csv
spamWriter = csv.writer(open('/home/madan/personal/development/SIMULATION_PART_4/data_processing/spectrometer_spaces_class.csv', 'wb'), delimiter=' ')
if __name__ == "__main__":
    f = open("/home/madan/personal/development/SIMULATION_PART_4/data_processing/spectrometer_spaces_processed.csv",'r')
    spamReader = csv.reader(f, delimiter=' ', quotechar='|')
    count = 0
    row_length = []
    data = []
    for row in spamReader:
        current_row = []
        for id, values in enumerate(row):
            if values:
                if id == 1:
                    old = int(values)
                    values  = int(values)/10
                    current_row.append(old)
                    print old
                    print values
                current_row.append(values)
        print current_row
        spamWriter.writerow(current_row)
        row_length.append(len(current_row))
        count += 1
    print row_length
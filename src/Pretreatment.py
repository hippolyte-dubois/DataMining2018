import csv


def training_csv(f_in="../data/training.csv", f_out="../data/training_treated.csv"):
    file_in = open(f_in,'r')
    file_out = open(f_out,"w+")
    reader = csv.reader(file_in, delimiter=';')
    writer = csv.writer(file_out, delimiter=';')
    for row in reader:
        row_out = []
        for v in row:
            v_out = v
            if not v:
                v_out = 0
            row_out.append(v_out)
        if row[0]!='Client':
            for n in [1,2,3]:
                if row_out[n]==0:
                    row_out[n]=False
                else:
                    row_out[n]=True
            try:
                row_out[4] = int(row_out[4])
            except:
                row_out[4] = 0
            row_out[5] = float(row_out[5])
            row_out[6] = float(row_out[6][1:-1])
            for n in row_out[7:]:
                n = float(n)
        writer.writerow(row_out)
        del row_out
    file_in.close()
    file_out.close()

def test_csv(f_in="../data/test.csv", f_out="../data/test_treated.csv"):
    file_in = open(f_in,'r')
    file_out = open(f_out,"w+")
    reader = csv.reader(file_in, delimiter=';')
    writer = csv.writer(file_out, delimiter=';')
    for row in reader:
        row_out = []
        for v in row:
            if row[0] != "NbSalaries":
                try:
                    v_out = float(v)
                except ValueError:
                    v_out = 0.0
            else:
                v_out = v
            row_out.append(v_out)
        writer.writerow(row_out)
        del row_out
    file_in.close()
    file_out.close()

test_csv()
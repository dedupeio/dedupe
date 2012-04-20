def canonicalImport(filename) :
    import csv

    data_d = {}
    with open(filename) as f :
        reader = csv.reader(f)
        header = reader.next()
        print header
        for i, row in enumerate(reader) :
            instance = {}
            for j, col in enumerate(row) :
                instance[header[j]] = col.strip()
            data_d[i] = instance

    return(data_d, header)
        


import sys

infile_name = str(sys.argv[1])
outfile_name = str(sys.argv[2])

with open(outfile_name, 'w') as outfile, open(infile_name, 'r', encoding='utf-8') as infile:
	for line in infile:
		numbers = line.split(",")
		for index, number in enumerate(numbers):
			if (index == 0):
				outfile.write(number)
				continue
			if float(number) != 0:
				outfile.write(" " + str(index) + ":" + number)
			if index == len(numbers) - 1:
				outfile.write("\n")
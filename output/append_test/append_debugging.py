	#f1 = open("file1_test.txt")
	#f1_contents = f1.read()
	f1.close()

	f2 = open("file2_test.txt")
	f2_contents =  f2.read()
	f2.close()

	f3 = open("combine.txt", "w")
	f3.write(f1_contents + str.rstrip('\n') + f2_contents)
	f3.close()
	
#combine_files_test()

#f1 = open("file1_test.txt", "r")
#f2 = open("file2_test.txt", "w")
#for line in f1:
	#f2.write("1" % line.replace("\n"))
#f1.close()
#f2.close()


combine()


	import ast
			
	file1 = open("file1_test.txt", "r+")
	file2 = open("file2_test.txt", "r+")
	
	file1Data = file1.read()
	file2Data = file2.read()
	
	file2Data = file2Data.split("\n")
	#file1Data = [file1Data[i].split() for i in range(len(file1Data))]
	
	#file2Data = ast.literal_eval(file2Data)
	
	#for i in range(len(file2Data)):
		#file1Data.append(file2Data[i])
		
	#file1.write(file2Data[i])

combine_files_test()


list in file2, split, for i in file2:
	
	#name1 = raw_input('file2_test.txt')
	#name2 = raw_input('file1_test.txt')
	
	fin = open('file2_test.txt', "r")
	data2 = fin.read()
	fin.close()
	
	file2 = file2.split("\n")
	#fout = open('file1_test.txt', "a")
	#fout = fout.split("\n")
	#for i in fout:
		#fout.write(data2)
	
	fout.close()
	
	
combine_files_test()


		lines = list(f)
		f.seek(0)
		for i in range(min(len(lines), len(alist))):
			f.write(lines[i].rstrip('\n')
			#f.write(' ')
			f.write(str(alist[i]))
			f.write('\n')

combine_files_test()
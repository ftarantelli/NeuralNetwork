CC = g++ -g -Wall #-I /home/users/tarante/armadillo-10.8.2/include/

dec.out: dec_Chebys.cpp
	$(CC) dec_Chebys.cpp -odec.out -larmadillo -llapack -lblas -fopenmp

rec.out: rec_Chebys.cpp
	$(CC) rec_Chebys.cpp -orec.out -larmadillo -llapack -lblas -fopenmp

bootdec.out: boot_dec_Chebys.cpp
	$(CC) boot_dec_Chebys.cpp -obootdec.out -larmadillo -llapack -lblas -fopenmp

ntk.out: main.cpp include/
	$(CC) main.cpp -ontk.out -larmadillo -llapack -lblas -fopenmp



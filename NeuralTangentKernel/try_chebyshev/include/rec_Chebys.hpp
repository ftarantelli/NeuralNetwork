#ifndef __REC_CHEB
#define __REC_CHEB

double a(1), b(10); // Estremi dell'intervallo

double approximatedFunction(const dvec& coefficients, double xvar, int N) {
    double result = 0.0;

    for (int n = 0; n <= N; ++n) {
        double yvar = 2./(b-a)*(xvar - a/2. - b/2.);
        result += coefficients(n) * chebyshev_t(n, yvar);
    }

    return result;
}

dvec rec_Chebys(const int N, const dvec aux_x) {
    //int N = 50; // Grado massimo della scomposizione
    double x; // Punto in cui calcolare f(x)

    dvec coefficients(N + 1); // Coefficienti a_n
    int dimm(size(aux_x).n_rows);
    dvec aux_f(dimm);
    std::ifstream input_file("temp.dat");
/*
	double value;
	int iiv(0);
	while( input_file >> value){
		coefficients(iiv) = value;
		++ iiv;
	}
*/
	for(int i=0; i < dimm; ++i){
        aux_f(i) = approximatedFunction(coefficients, aux_x(i), N);

        std::cout << "f(" << x << ") = " << result << std::endl;
    }
	input_file.close();
    return 0;
}

#endif

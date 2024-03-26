#include <iostream>
#include <armadillo>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/chebyshev.hpp>
#include <filesystem>

namespace fs = std::filesystem;

using namespace arma;
using namespace boost::math;


double discr_integration(const int n_cheb, const dvec axis, const dvec func){

	int dimen = size(func).n_rows;
	int dim_axis = size(func).n_rows;

	if(dimen != dim_axis)
		std::cerr << "Mismatch of axis and function\n";

	// Multiplication between chebyshev and function
	double a = axis(0), b = axis(dimen-1);

	auto function = [n_cheb,a,b,dimen,axis,func](double xvar) {
		for(int ind=1; ind < dimen; ++ind)
			if(xvar >= axis(ind-1) && xvar < axis(ind)){
				double yvar = 2./(b-a)*(xvar - a/2. - b/2.);
				double yyfunc = (func(ind)-func(ind-1))/(axis(ind) - axis(ind-1))*(xvar-axis(ind)) + func(ind);
				return chebyshev_t(n_cheb, yvar) / sqrt(1.0 - std::pow(yvar,2.)) * 2./(b-a)*yyfunc;
			};
	};

	// Perform the numerical integration
	//integrator.integrate(function, a, b, integral, );
	double result = quadrature::gauss_kronrod<long double,61>::integrate(function,a,b,5,1e-16);

    return result;

}

int main(void) {
    int N_corr = 12; // Grado massimo della scomposizione
    double coeff; // Estremi dell'intervallo
    double integral(0.), error(0.);
	const int max_iterations = 15;

    //dvec coefficients(N + 1, fill::zeros); // Vettore dei coefficienti

	fs::path stringInput = "bootstrap/bootstrap1k_mean_secondofile.dat";
    std::ifstream inputFile(stringInput);

    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the input file." << std::endl;
    }

    dvec boot, time, corr;

    std::string line;
    while (std::getline(inputFile, line)) {
        // Check if the line starts with '#', indicating a comment
        if (line.empty() || line[0] == '#') {
            continue; // Skip comments
        }

        std::istringstream iss(line);

        double val1, val2, val3;

        while (iss >> val1 >> val2 >> val3) {
            boot.resize(boot.n_elem + 1);
            time.resize(time.n_elem + 1);
            corr.resize(corr.n_elem + 1);

            boot(boot.n_elem - 1) = val1;
            time(time.n_elem - 1) = val2;
            corr(corr.n_elem - 1) = val3;
        }
    }
    inputFile.close();
//std::cout << "aaaaa1\n";
    int time_max = int(max(time) + 1);
    int harm_boot_max = int(max(boot));

	dvec time_corr = linspace(0,99,time_max);

			std::string outputFileName = (stringInput).string(), substringToRemove = "bootstrap/";
			size_t pos = outputFileName.find(substringToRemove);
//std::cout << outputFileName << "  aaaaa1\n";
			outputFileName.erase(pos, substringToRemove.length());
//std::cout << "aaaaa2\n";
			outputFileName = "bootstrap/decCheb_" + outputFileName;
			std::ofstream outputFile(outputFileName);
//std::cout << outputFileName << std::endl;
			for (int n = 0; n <= N_corr; ++n) {

				// Choose an appropriate quadrature rule (e.g., 15-point Gauss-Kronrod)

				quadrature::gauss_kronrod<double, max_iterations> integrator;

				integral = discr_integration(n, time_corr, corr);
				coeff = (2.0 / M_PI) * integral;
				if(n==0) coeff *= 0.5;

				outputFile << harm_boot_max << "\t\t" << n << "\t\t" << coeff << std::endl;

			}

//std::cout << outputFileName << coeff << std::endl;
			// Close the output file
			outputFile.close();



    return(0);
}

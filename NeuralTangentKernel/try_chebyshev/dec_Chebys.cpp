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
			}
	};

	// Perform the numerical integration
	//integrator.integrate(function, a, b, integral, );
	double result = quadrature::gauss_kronrod<long double,61>::integrate(function,a,b,15,1e-16);

/*
	// Computation of the integral
	double result = 0.0;

    for (int i = 1; i < dimen-2; ++i) {
        double h = axis(i+1) - axis(i);
        result += 0.5 * (yfunc(i-1) + yfunc(i)) * h;
    }

    //std::cout << result << "\n";
*/
    return result;

}

int main(void) {
    int N_corr = 30, N_rho = 30; // Grado massimo della scomposizione
    double coeff; // Estremi dell'intervallo
    double integral(0.), error(0.);
	const int max_iterations = 15;

    //dvec coefficients(N + 1, fill::zeros); // Vettore dei coefficienti

	int num_sample = 1, time = 12;
	int Ntotdata = num_sample + time;

	dvec time_corr = linspace(1,12,time);
	dvec w_rho = linspace(0., 0.3, num_sample);

	fs::path folderPath = "fakedata/";

	if (!fs::is_directory(folderPath)) {
        std::cerr << "The folder does not exist." << std::endl;
    }

    int index(0);
    // Iterate through all files in the folder
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        // Check if the file is a regular text file
        if (entry.is_regular_file() && entry.path().filename().string().find("decCheb_") == std::string::npos) {
        //if (entry.is_regular_file()) {

            std::ifstream file(entry.path());
//std::cout << entry.path() << '\n';
            // Check if the file can be opened
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << entry.path() << std::endl;
                continue;
            }

            dvec corr(time), rho(num_sample);

            double number;
            int ind_read(0.);
            dvec xx(Ntotdata);
            while (file >> number) {
               xx(ind_read) = number;
               ++ind_read;
            }

            corr = xx.subvec(0, time-1);
            rho = xx.subvec(Ntotdata-num_sample, Ntotdata-1);

            // Close the file
            file.close();
            ++index;

			std::string outputFileName = (entry.path()).string(), substringToRemove = "fakedata/";
			size_t pos = outputFileName.find(substringToRemove);

			outputFileName.erase(pos, substringToRemove.length());

			outputFileName = "fakedata/decCheb_" + outputFileName;
			std::ofstream outputFile(outputFileName);
std::cout << time_corr;
			for (int n = 0; n <= N_corr; ++n) {

				// Choose an appropriate quadrature rule (e.g., 15-point Gauss-Kronrod)

				quadrature::gauss_kronrod<double, max_iterations> integrator;

				integral = discr_integration(n, time_corr, corr);
				coeff = (2.0 / M_PI) * integral;
				if(n==0) coeff *= 0.5;

				outputFile << coeff << std::endl;

			}

			outputFile << std::endl;
//std::cout << outputFileName << coeff << std::endl;
			for (int n = 0; n <= N_rho; ++n) {

				// Choose an appropriate quadrature rule (e.g., 15-point Gauss-Kronrod)

				quadrature::gauss_kronrod<double, max_iterations> integrator;

				integral = discr_integration(n, w_rho, rho);
				coeff = (2.0 / M_PI) * integral;
				if(n==0) coeff *= 0.5;

				outputFile << coeff << std::endl;

			}
//std::cout << outputFileName << coeff << std::endl;
			// Close the output file
			outputFile.close();
        }
    }



/*
	for(int ii=0; ii<10; ++ii){
		xx(ii) = ii + 1.;
		yy(ii) = xx(ii)*xx(ii);
	}


	a = xx(0); b = xx(9);

    for (int n = 0; n <= N; ++n) {

		// Choose an appropriate quadrature rule (e.g., 15-point Gauss-Kronrod)

		quadrature::gauss_kronrod<double, max_iterations> integrator;
*/
/*
		auto function = [n,a,b](double xvar) {
			double yvar = 2./(b-a)*(xvar - a/2. - b/2.);
			return chebyshev_t(n, yvar) / sqrt(1.0 - std::pow(yvar,2.)) * 2./(b-a) *(xvar*xvar);
		};

		// Perform the numerical integration
		//integrator.integrate(function, a, b, integral, );
		integral = quadrature::gauss_kronrod<long double,61>::integrate(function,a,b,15,1e-16);
		//std::cout << integral << std::endl;
        coefficients(n) = (2.0 / M_PI) * integral;
		if(n==0) coefficients(n) *= 0.5;
*/


/*
		integral = discr_integration(n, xx, yy);
		coefficients(n) = (2.0 / M_PI) * integral;
		if(n==0) coefficients(n) *= 0.5;

	}
*/



    // Ora hai calcolato i coefficienti a_n

    // Puoi utilizzare questi coefficienti per stimare f(x) utilizzando la scomposizione
	//std::cout.precision(9);
	//std::cout << coefficients << std::endl;

    return 0;
}

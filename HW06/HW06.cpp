#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <time.h>
#include </opt/homebrew/Cellar/open-mpi/4.1.5/include/mpi.h>

// Initial Setting
const int N     = 16;  // number of computing cells
const int ROW   = N;   // it must be equal to N (grids)
const int COL   = N;   // it must be equal to N (grids)

const double pi     = M_PI;
const int iter_max  = 1000;

const double tol    = 1e-7; // tol: tolerance

// 2D Poisson Equation: Constants
const double L      = 1.0;  // 1-D computational domain size
const double u0     = 0.0;  // background density
const double amp    = 1.0;  // sinusoidal amplitude
const double cfl    = 1.0;  // Courant condition factor


int main (void){
    void SOR_method(double **u, double **u_in, double **rho, double dx);
    void ref_func(double **M, double dx, double dy);
    void init_rho(double **rho, double dx, double dy);
    void init_u(double **u);
    double calculateError(double **u, double **u_ref);
    void save_M(double **M, const std::string& filename);


    //-----------------------------
    //  2D Poisson Equation
    //-----------------------------
    double **u, **u_in, **u_ref, **rho;

    // Derived constants
    double dx = L / (N - 1); // spatial resolution
    double dy = L / (N - 1); // spatial resolution

    u     = (double **) malloc(ROW * sizeof(double));
    u_in  = (double **) malloc(ROW * sizeof(double));
    u_ref = (double **) malloc(ROW * sizeof(double));
    rho   = (double **) malloc(ROW * sizeof(double));
    for (int i=0;i<ROW;i++){
        u[i]      = (double *) malloc(COL * sizeof(double));
        u_in[i]   = (double *) malloc(COL * sizeof(double));
        u_ref[i]  = (double *) malloc(COL * sizeof(double));
        rho[i]    = (double *) malloc(COL * sizeof(double));        
    }    

    // Initialize rho and u values
    init_rho(rho, dx, dy);
    init_u(u);

    // Compute u_ref values
    ref_func(u_ref, dx, dy);

    //-----------------------------
    //  Poisson Solver
    //-----------------------------
    double err;

    SOR_method(u, u_in, rho, dx);
    err = calculateError(u, u_ref);
    for (int i = 0; i < ROW; i++){
        for (int j = 0; j < COL; j++){
            printf(" %f ", u[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < ROW; i++){
        for (int j = 0; j < COL; j++){
            printf(" %f ", u_ref[i][j]);
        }
        printf("\n");
    }
    printf("Error: %f \n", err);            
    // save array u
    save_M(u, "./SOR_result.txt");

    // free memory
    free(u);
    free(u_in);
    free(u_ref);
    free(rho);

    return 0;    
}

//--------------------
//  SOR Method
//--------------------
void SOR_method(double **u, double **u_in, double **rho, double dx){
    // Initialize variables
    // Parameters of wall-clock time
    double start = omp_get_wtime();

    // Time stepping loop
    for (int count=0; count<iter_max; count++) {
        // Backup the input data
        u_in = u;

        double err = 0.0;
        double w = 1.8; // omega

        // Update all cells

        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                int mod = (i+j)%2;
                if (mod == 0){
                    double res = (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] \
                        - 4 * u_in[i][j] - dx * dx * rho[i][j]) / (dx * dx);
                    u[i][j] = u_in[i][j] + w * dx * dx * res / 4.0;
                    err += dx * dx * abs(res / u[i][j]) / (N * N);                    
                }
                else if (mod == 1){
                    double res = (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] \
                        - 4 * u_in[i][j] - dx * dx * rho[i][j]) / (dx * dx);
                    u[i][j] = u_in[i][j] + w * dx * dx * res / 4.0;
                    err += dx * dx * abs(res / u[i][j]) / (N * N);                    
                }
            }
        }

        // Boundary conditions
        for (int j=0; j<COL; j++){
            u[0][j] = 0;
            u[ROW-1][j] = 0;
        }

        for (int i=0; i<ROW; i++) {
            u[i][0] = u[i][1];
            u[i][COL-1] = u[i][COL-2];
        }

        // Update time
        count++;
        double end = omp_get_wtime();

        if (err < tol) {
            double diff = ((double)(end - start)); // CLOCKS_PER_SEC;
            printf("wall-clock time: %f \n", diff);
            printf("iterations: %d \n", count);
            break;
        }
    }
}

//---------------------
//  Setting
//---------------------

// Define a reference analytical solution
void ref_func(double **M, double dx, double dy) {
    double k = 2.0 * pi / L;                      // wavenumber
    for (int i=0; i<ROW; i++){
        for (int j=0; j<COL; j++){
            M[i][j] = sin(pi*i*dx) * cos(pi*j*dy);
        }
    }
}

// Define an initial density distribution: f(x,y)
void init_rho(double **rho, double dx, double dy) {
    double k = 2.0 * pi / L;
    for (int i=0; i<ROW; i++){
        for (int j=0; j<COL; j++){
            rho[i][j] = -1/(2*pow(pi,2))*sin(pi*i*dx)*cos(pi*j*dy);
        }
    }
}

//---------------------
//  Useful Tools
//---------------------

// Initialize u values to 0: unknown u(x,y)
void init_u(double **u) {
    for (int i=0; i<ROW; i++){
        for (int j=0; j<COL; j++){
            u[i][j] = 0;
        }
    }
}

// Function to calculate the error between u and u_ref
double calculateError(double **u, double **u_ref) {
    double error = 0.0;
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            error += abs(u[i][j] - u_ref[i][j]);
        }
    }
    return error / (N * N);
}

// Save the array to the text file
void save_M(double **M, const std::string& filename){
    FILE* file = fopen(filename.c_str(), "w");

    if (file != nullptr){
        for (int i = 0; i < ROW; i++){
            for (int j = 0; j < COL; j++){
                fprintf(file, " %f ", M[i][j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        printf("Array data saved to file \n");
    }
    else{
        printf("Unable to open file \n");
        printf("(can't find the TXT file in your folder 'RESULT', \n");
        printf("try to create TXT file and check the variable 'fileroot' in main.cpp!) \n");
    }
}
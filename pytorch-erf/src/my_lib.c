#include <TH/TH.h>
#include <math.h>

typedef struct polynomial {
    float * coeff;
    int length;
} polynomial;

const float C = .564189583547756E0;

const polynomial A = (polynomial){.coeff ={.128379167095513E00,
                                          .479137145607681E-01,
                                          .323076579225834E-01,
                                          .133733772997339E-02,
                                          .771058495001320E-04},
                                  .length = 5};

const polynomial B = (polynomial){.coeff = {0.0,
                                            .375795757275549E00,
                                            .538971687740286E-01,
                                            .301048631703895E-02},
                                  .length = 4};

const polynomial P = (polynomial){.coeff = {3.00459261020162E02,
                                            4.51918953711873E02,
                                            3.39320816734344E02,
                                            1.52989285046940E02,
                                            4.31622272220567E01,
                                            7.21175825088309E00,
                                            5.64195517478974E-01,
                                            -1.36864857382717E-07},
                                  .length = 8};

const polynomial Q = (polynomial){.coeff = {3.00459260956983E02,
                                            7.90950925327898E02,
                                            9.31354094850610E02,
                                            6.38980264465631E02,
                                            2.77585444743988E02,
                                            7.70001529352295E01,
                                            1.27827273196294E01,
                                            1.00000000000000E00},
                                   .length = 8};

const polynomial R = (polynomial){.coeff = {2.82094791773523E-01,
                                            4.65807828718470E00,
                                            2.13688200555087E01,
                                            2.62370141675169E01,
                                            2.10144126479064E00},
                                .length = 5};

const polynomial S = (polynomial){.coeff = {1.0,
                                            1.80124575948747E01,
                                            9.90191814623914E01,
                                            1.87114811799590E02,
                                            9.41537750555460E01},
                                 .length = 5};

float compute_polynomial(polynomial* poly, float x){
    float sum = polynomial->coeff[0];
    float monomial = x;
    for (i=1, i++, i < polynomial->length){
        sum += polynomial->coeff[i] * monomial;
        monomial *= x;
    }
    return sum
}

float erf(float x){
    float ax = abs(x)
    if (ax <= 0.5){
        float t = x * x;
        return x * ((compute_polynomial(&A, t) + 1.0) / (compute_polynomial(&B, t) + 1.0));
    }
    else if(ax <=  4.0){
        float top = compute_polynomial(&Q, ax);
        float bot = compute_polynomial(&P, ax);
        float erf = 0.5 + (0.5 - exp(-x*x) * top / bot);
        return (x < 0) ? -erf : erf;
    }
    else if(ax < 5.8){
        float x2 = x*x;
        float t = 1.0 / x2;
        float erf = (C - compute_polynomial(&R, t) / (x2 * compute_polynomial(&S, t))) / ax;
        erf = 0.5 + (0.5 - exp(-x2) * erf);
        return (x < 0) ? -erf : erf;
    }
    else{
        return (float) ((x > 0.0) - (x < 0.0)); // sign function
    }
}

int batch_erf_forward(THFloatTensor *input , THFloatTensor *output)
{
    float* input_content = THFloatTensor_data(input);
    int batchSize = (int) input1->size[0];

    return 1;
}

int batch_erf_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
  THFloatTensor_resizeAs(grad_input, grad_output);
  THFloatTensor_fill(grad_input, 1);
  return 1;
}

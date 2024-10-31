using MathNet.Numerics;
using MathNet.Numerics.OdeSolvers;
using System;
using System.Collections.Generic;

using MathNet.Numerics.LinearAlgebra;
using System.Data;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.RootFinding;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Collections;
using ScottPlot;
class Program
{
    static void Main()
    {
        double t = 0;
        double ta = 0.5;
        double tf = 600;
        int it = 0;
        double array_size = (tf-t)/ta;
        Vector<double> y;

        double[] results = new double[(int)array_size + 1];
        double[] tempos = new double[(int)array_size + 1];
        
        var u = Vector<double>.Build.Dense(new double[] { 16.6, 0.55, 15.6, 0 });
        var x0 = Vector<double>.Build.Dense(new double[] { 14, -4.32e-4, 5.28e-4, 0 });

        while (t <= tf)
        {
            if (t == 200.0)
                u[2] = u[2]*0.90;

            y = phModel(u: u, x0: x0, t: t, ta: ta);

            results[it] = y[3];
            tempos[it] = t;

            t += ta;
            it += 1;
        }

        ScottPlot.Plot myPlot = new();

        myPlot.Add.Scatter(tempos, results);
        myPlot.SavePng("ph1.1.png", 400, 300);
    }
    static Vector<double> phModel(Vector<double> u, Vector<double> x0, double t, double ta)
    {
        double pk1 = 6.35;
        double pk2 = 10.25;
        double q1 = u[0];
        double q2 = u[1];
        double q3 = u[2];

        var ode_out = ModifiedRungeKuttaFourthOrder(x0, 0, t+ta, N: 100, (t, y) => dxdt(t, y, q1, q2, q3));

        Vector<double> x = Vector<double>.Build.Dense(new double[] {
            ode_out[ode_out.Length - 1][0],
            ode_out[ode_out.Length - 1][1],
            ode_out[ode_out.Length - 1][2],
            ode_out[ode_out.Length - 1][3]}
        );

        var y = x;

        Func<double, double> ph_func = z =>
        {
            return y[1] + Math.Pow(10, (z - 14)) + y[2] * ((1 + 2 * Math.Pow(10, (z - pk2))) / (1 + Math.Pow(10, (pk1 - z)) + Math.Pow(10, (z - pk2)))) - Math.Pow(10, -z);
        };

        double ph_out = Bisection.FindRoot(ph_func, lowerBound: 0, upperBound: 14, accuracy: 1e-9);

        y[3] = ph_out;

        return y;
    }
    public static Vector<double>[] ModifiedRungeKuttaFourthOrder(Vector<double> y, double start, double end, int N, Func<double, Vector<double>, Vector<double>> f)
    {
        double num = (end - start) / (double)(N - 1);
        double num2 = start;
        var vectors = new Vector<double>[N];

        for (int i = 0; i < N; i++)
        {
            Vector<double> num3 = f(num2, y);
            Vector<double> num4 = f(num2 + num / 2.0, y + num3 * num / 2.0);
            Vector<double> num5 = f(num2 + num / 2.0, y + num4 * num / 2.0);
            Vector<double> num6 = f(num2 + num, y + num5 * num);

            vectors[i] = y + num / 6.0 * (num3 + 2.0 * num4 + 2.0 * num5 + num6);
            num2 += num;
            y = vectors[i];
        }
        return vectors;
    }
    static Vector<double> dxdt(double t, Vector<double> y, double q1, double q2, double q3)
    {
        double h, wa4, wb4, ph_out;

        double A = 207;
        double Cv = 8.75;

        double wa1 = 3e-3;
        double wa2 = -3e-2;
        double wa3 = -3.05e-3;

        double wb1 = 0;
        double wb2 = 3e-2;
        double wb3 = 5e-5;

        h = y[0];
        wa4 = y[1];
        wb4 = y[2];
        ph_out = y[3];


        double dh = 1 / A * (q1 + q2 + q3 - Cv * (Math.Sqrt(h)));
        double dwa4 = 1 / (A * h) * ((wa1 - wa4) * q1 + (wa2 - wa4) * q2 + (wa3 - wa4) * q3);
        double dwb4 = 1 / (A * h) * ((wb1 - wb4) * q1 + (wb2 - wb4) * q2 + (wb3 - wb4) * q3);

        var _dxdt = Vector<double>.Build.Dense(new double[] { dh, dwa4, dwb4, ph_out });

        return _dxdt;
    }
}
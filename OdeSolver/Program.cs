using MathNet.Numerics.OdeSolvers;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.RootFinding;
class Program
{
    static void Main()
    {
        double t = 0;
        double ta = 5;
        double tf = 600;
        int it = 0;
        double array_size = (tf-t)/ta;
        Vector<double> y;

        double[] y0 = new double[(int)array_size + 1];
        double[] y1 = new double[(int)array_size + 1];
        double[] y2 = new double[(int)array_size + 1];
        double[] y3 = new double[(int)array_size + 1];
        double[] times = new double[(int)array_size + 1];
        
        var u = Vector<double>.Build.Dense(new double[] { 16.6, 0.55, 15.6, 0 });
        var x0 = Vector<double>.Build.Dense(new double[] { 14, -4.32e-4, 5.28e-4, 0 });
        var y_in = x0;

        while (t <= tf)
        {
            // step at t = 200 s
            //if (t == 200.0)
            //    u[2] = u[2]*1.05;

            y = phModel(u: u, y_in : y_in, t: t, ta: ta);

            //plot vectors assignment
            y0[it] = y[0];
            y1[it] = y[1];
            y2[it] = y[2];
            y3[it] = y[3];
            times[it] = t;

            // simulation update time/iteration variables
            t += ta;
            it += 1;
            y_in = y;
        }

        ScottPlot.Multiplot multiplot = new();

        ScottPlot.Plot plot1 = new();
        ScottPlot.Plot plot2 = new();
        ScottPlot.Plot plot3 = new();
        ScottPlot.Plot plot4 = new();

        plot1.Add.Scatter(times, y0);
        plot2.Add.Scatter(times, y1);
        plot3.Add.Scatter(times, y2);
        plot4.Add.Scatter(times, y3);

        plot1.Title("h");
        plot2.Title("wa4");
        plot3.Title("wb4");
        plot4.Title("pH");

        multiplot.AddPlot(plot1);
        multiplot.AddPlot(plot2);
        multiplot.AddPlot(plot3);
        multiplot.AddPlot(plot4);

        multiplot.Layout = new ScottPlot.MultiplotLayouts.Columns();

        multiplot.SavePng("q3.png", 800, 600);
    }
    static Vector<double> phModel(Vector<double> u, Vector<double> y_in, double t, double ta)
    {
        double pk1 = 6.35;
        double pk2 = 10.25;

        double q1 = u[0];
        double q2 = u[1];
        double q3 = u[2];

        var ode_out = RungeKutta.FourthOrder(y_in, t, t + ta, N : 100, (t, y_in) => dxdt(t, y_in, q1, q2, q3));

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

        y[3] = Bisection.FindRoot(ph_func, lowerBound: 0, upperBound: 14, accuracy: 1e-9);

        return y;
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

        var out_dxdt = Vector<double>.Build.Dense(new double[] { dh, dwa4, dwb4, ph_out });

        return out_dxdt;
    }
}
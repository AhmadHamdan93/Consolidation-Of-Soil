using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANNtrainingbyABC
{
    internal class Bee
    {
        double[,] population;
        int food;
        int iteration;
        int limit;
        int D;
        int[] trail;
        double[] probability;
        double[] OptimumSol;
        Random rand;
        int ub;
        int lb;
        // save some data from neural network for evaluate solution
        double[][] trainingData;
        double[] outputTrainingData;
        //------------------------
        //--------for calc MAE Error ----------------
        //double MAE_Error;   //for save optimum MAE Error
        double[] MAE_Errors;    // for save levels of Error when every iteration
        double[] RMSE_Errors;   // for save levels of Error when every iteration
        //------------------------
        //------------------------
        double[][,] Weight; // for know dimentions array of data
        Neuron[][] Neurons;

        public Bee(double[,] solutions, double[][] input, double[] y, double[][,] W, Neuron[][] N, int Epocs)
        {
            food = solutions.GetLength(0);
            D = solutions.GetLength(1) - 1;
            iteration = Epocs; //1000; //300;
            //limit = food * D ;
            limit = 50;
            trail = new int[food];
            population = solutions;
            probability = new double[food];
            OptimumSol = new double[D + 1];
            rand = new Random();
            ub = Convert.ToInt32(findMaxValue(solutions));
            lb = Convert.ToInt32(findMinValue(solutions));
            trainingData = input;
            outputTrainingData = y;
            Weight = W;
            Neurons = N;
            // ----------------
            //MAE_Error = 0.0;
            MAE_Errors = new double[Epocs];
            RMSE_Errors = new double[Epocs];
        }

        public double[] getBestSolution()
        {
            return OptimumSol;
        }

        public double[][,] convert1D_to_3D(double[] a)
        {
            // here we must already know size 3D array in every level
            // for reason we based on 'w' array for know it
            double[][,] array = new double[Weight.GetLength(0)][,];
            for (int i = 0; i < Weight.GetLength(0); i++)
                array[i] = new double[Weight[i].GetLength(0), Weight[i].GetLength(1)];

            // ----------------------------
            int t = 0;
            for (int i = 0; i < Weight.GetLength(0); i++)
                for (int j = 0; j < Weight[i].GetLength(0); j++)
                    for (int k = 0; k < Weight[i].GetLength(1); k++)
                    {
                        array[i][j, k] = a[t];
                        t++;
                    }
            return array;
        }

        public double findMinValue(double[,] solutions)
        {
            double min = solutions[0, 0];
            for (int i = 0; i < food; i++)
                for (int j = 0; j < D; j++)
                    if (min > solutions[i, j])
                        min = solutions[i, j];
            return min;
        }

        public double findMaxValue(double[,] solutions)
        {
            double max = solutions[0, 0];
            for (int i = 0; i < food; i++)
                for (int j = 0; j < D; j++)
                    if (max < solutions[i, j])
                        max = solutions[i, j];
            return max;
        }

        public void Search()
        {
            Console.WriteLine();
            //-----------------------
            SaveOptimumSol(0);
            SelectOptimumSolution();
            for (int iter = 0; iter < iteration; iter++)
            {
                WorkerBee();
                findProbability();
                LookerBee();
                SelectOptimumSolution();
                ScouterBee();

                // save array of error
                MAE_Errors[iter] = MAE_Calc(OptimumSol); // MAE_Calc(OptimumSol);OptimumSol[D]
                RMSE_Errors[iter] = OptimumSol[D];//RMSE_Calc(OptimumSol); // OptimumSol[D];

            }
            // here we should find mae error
            // by function is MAE_Calc
            //MAE_Error = MAE_Calc(OptimumSol);


            Console.WriteLine("'RMSE' Root Mean Square Error rate after apply ABC algorithm is: " + RMSE_Errors[RMSE_Errors.Length - 1]);
            Console.WriteLine("'MAE' The Mean Absolute Error rate after apply ABC algorithm is: " + OptimumSol[D]);

        }

        double RMSE_Calc(double[] sol)
        {
            // this function work to find RMSE Error 

            double[][,] tempWeights = convert1D_to_3D(sol);
            double result = 0.0;
            int lenData = trainingData.GetLength(0);
            int d = trainingData[0].GetLength(0);
            for (int i = 0; i < lenData; i++)
            {
                double[] rowData = new double[trainingData[0].GetLength(0)];
                // fetch row from training data then send it to predict
                rowData = getRowFromArray(trainingData, i);
                // calculate output from predict
                double[] outputPredict = Predict(rowData, tempWeights);
                // calculate error 
                result += Math.Abs((outputPredict[0] - outputTrainingData[i])) * (outputPredict[0] - outputTrainingData[i]);
            }
            result = result / lenData;
            result = Math.Sqrt(result);
            return result;
        }

        double MAE_Calc(double[] sol)
        {
            // this function work to find MAE Error 

            double[][,] tempWeights = convert1D_to_3D(sol);
            double result = 0.0;
            int lenData = trainingData.GetLength(0);
            int d = trainingData[0].GetLength(0);
            for (int i = 0; i < lenData; i++)
            {
                double[] rowData = new double[trainingData[0].GetLength(0)];
                // fetch row from training data then send it to predict
                rowData = getRowFromArray(trainingData, i);
                // calculate output from predict
                double[] outputPredict = Predict(rowData, tempWeights);
                // calculate error 
                result += Math.Abs((outputPredict[0] - outputTrainingData[i]));//* (outputPredict[0] - outputTrainingData[i]);
            }
            result = result / lenData;
            //result = Math.Sqrt(result);
            return result;
        }

        public void LookerBee()
        {
            int i, t;
            t = 0;

            /*onlooker Bee Phase*/
            while (t < food)
            {
                i = findMaxProb(probability);
                t++;
                //--------------------------

                int neighbor = rand.Next(food);
                while (neighbor == i)
                {
                    neighbor = rand.Next(food);
                }
                // detect current sol and new sol
                double[] newRowSolution = LocalSearch(i, neighbor);
                if (newRowSolution[D] < population[i, D])
                {
                    replaceRow(newRowSolution, i);
                    trail[i] = 0;
                }
                else
                {
                    trail[i] += 1;
                }

            }/*while*/

            /*end of onlooker bee phase     */
        }

        public int findMaxProb(double[] arr)
        {
            double r = rand.NextDouble();
            //double max = arr[0];
            int idx = 0;
            for (int i = 0; i < arr.Length; i++)
            {
                if (r < arr[i])
                    return i;
            }
            return idx;
        }

        public void WorkerBee()
        {
            /*start of employed bee phase*/
            for (int i = 0; i < food; i++)
            {
                int neighbor = rand.Next(food);
                while (neighbor == i)
                {
                    neighbor = rand.Next(food);
                }
                // detect current sol and new sol
                double[] newRowSolution = LocalSearch(i, neighbor);
                if (newRowSolution[D] < population[i, D])
                {
                    replaceRow(newRowSolution, i);
                    trail[i] = 0;
                }
                else
                {
                    trail[i] += 1;
                }
            }
            /*end of employed bee phase*/
        }

        public void replaceRow(double[] sol, int idx)
        {
            for (int j = 0; j < D + 1; j++)
            {
                population[idx, j] = sol[j];
            }
        }

        public double[] LocalSearch(int currentIDX, int neighbor)
        {
            double[] newSol = new double[D + 1];
            for (int i = 0; i < D; i++)
            {
                double randomNumber = lb + rand.NextDouble() * (ub - lb);
                newSol[i] = population[currentIDX, i] + randomNumber * (population[currentIDX, i] - population[neighbor, i]);
            }
            // Evaluate solution
            newSol[D] = Evaluate(newSol);
            return newSol;
        }

        public void ScouterBee()
        {
            for (int i = 0; i < food; i++)
            {
                if (trail[i] >= limit)
                {
                    initSol(i);
                    trail[i] = 0;
                    // evaluate new solution and save it
                    // Evaluate solution
                    double[] row = fetchRow(i);
                    population[i, D] = Evaluate(row);
                }
            }
        }

        public double[] fetchRow(int idx)
        {
            double[] row = new double[D + 1];
            for (int j = 0; j < D + 1; j++)
            {
                row[j] = population[idx, j];
            }
            return row;
        }

        public void initSol(int idx)
        {
            for (int i = 0; i < D; i++)
            {
                population[idx, i] = lb + rand.NextDouble() * (ub - lb);
            }
        }

        public void SelectOptimumSolution()
        {
            double minFx = population[0, D];
            int minIndex = 0;
            if (minFx < OptimumSol[D])
                SaveOptimumSol(minIndex);
            for (int i = 1; i < food; i++)
            {
                if (population[i, D] < minFx)
                {
                    minFx = population[i, D];
                    minIndex = i;
                    if (minFx < OptimumSol[D])
                        SaveOptimumSol(minIndex);
                }
            }
        }

        public void SaveOptimumSol(int idx)
        {
            for (int i = 0; i < D + 1; i++)
            {
                OptimumSol[i] = population[idx, i];
            }
        }

        public void findProbability()
        {
            double sumFX = 0.0;
            for (int i = 0; i < food; i++)
            {
                sumFX += population[i, D];
            }
            //---------------testing for not show sum == 0.0---------------
            if (sumFX == 0.0)
            {
                sumFX = 1;
            }
            //-----------------------------
            probability[0] = population[0, D] / sumFX;
            for (int i = 1; i < food; i++)
            {
                probability[i] = probability[i - 1] + population[i, D] / sumFX;
            }
        }

        public double Evaluate(double[] sol)
        {
            // this function work to find RMSE Error 

            double[][,] tempWeights = convert1D_to_3D(sol);
            double result = 0.0;
            int lenData = trainingData.GetLength(0);
            int d = trainingData[0].GetLength(0);
            for (int i = 0; i < lenData; i++)
            {
                double[] rowData = new double[trainingData[0].GetLength(0)];
                // fetch row from training data then send it to predict
                rowData = getRowFromArray(trainingData, i);
                // calculate output from predict
                double[] outputPredict = Predict(rowData, tempWeights);
                // calculate error 
                result += (outputPredict[0] - outputTrainingData[i]) * (outputPredict[0] - outputTrainingData[i]);
            }
            result = result / lenData;
            result = Math.Sqrt(result); // for RMSE
            return result;
        }

        public double[] getRowFromArray(double[][] dataArray, int idx)
        {
            double[] row = new double[dataArray[0].GetLength(0)];
            for (int i = 0; i < dataArray[0].GetLength(0); i++)
            {
                row[i] = dataArray[idx][i];
            }
            return row;
        }

        public double Fx(double value)
        {
            return value * value;
        }

        public double[] Predict(double[] input, double[][,] currentWeight)
        {
            //Forward propagation
            //Fill the first layers output neurons with input data
            for (int d = 0; d < Neurons[0].Length; d++)
                Neurons[0][d].output = input[d];

            //Feed forward phase
            for (int l = 1; l < Neurons.GetLength(0); l++)
            {
                //Now compute layer l, n is each neuron in layer l
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    //Compute neuron n in layer l
                    double sum = 0;

                    //Iterate over previous layers outputs and weights
                    //j is each of the previous layers neuron
                    for (int j = 0; j < Neurons[l - 1].Length; j++)
                        sum += (Neurons[l - 1][j].output * currentWeight[l - 1][j, n]);

                    //Store the weighted inputs on input.
                    Neurons[l][n].input = sum;

                    //The output is the sigmoid of the weighted input 
                    Neurons[l][n].output = Sigmoid(Neurons[l][n].input + 1);
                }
            }

            //prepare a vector of outputs to return
            var outputlayer = Neurons.GetLength(0) - 1;
            var output = new double[Neurons[outputlayer].Length];
            for (int n = 0; n < output.Length; n++)
                output[n] = Neurons[outputlayer][n].output;

            return output;
        }

        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double[] getRmse(int maximum)
        {
            int period = (iteration / maximum);
            double[] rmse = new double[maximum];
            for (int i = 0; i < maximum; i++)
            {
                int idx = i * period;
                rmse[i] = RMSE_Errors[idx];
            }
            return rmse;
        }

        public double[] getMae(int maximum)
        {
            int period = (iteration / maximum);
            double[] mae = new double[maximum];
            for (int i = 0; i < maximum; i++)
            {
                int idx = i * period;
                mae[i] = MAE_Errors[idx];
            }
            return mae;
        }
        // ==========================================
        // ==========================================
        // ==========================================
        //double[,] population;
        //int food;
        //int iteration;
        //int limit;
        //int D;
        //int[] trail;
        //double[] probability;
        //double[] OptimumSol;
        //Random rand;
        //int ub;
        //int lb;
        //// save some data from neural network for evaluate solution
        //double[][] trainingData;
        //double[] outputTrainingData;
        ////------------------------
        ////--------for calc MAE Error ----------------
        ////double MAE_Error;   //for save optimum MAE Error
        //double[] MAE_Errors;    // for save levels of Error when every iteration
        //double[] RMSE_Errors;   // for save levels of Error when every iteration
        ////------------------------
        ////------------------------
        //double[][,] Weight; // for know dimentions array of data
        //Neuron[][] Neurons;

        //public Bee(double[,] solutions, double[][] input, double[] y, double[][,] W, Neuron[][] N, int Epocs)
        //{
        //    food = solutions.GetLength(0);
        //    D = solutions.GetLength(1) - 1;
        //    iteration = Epocs; //1000; //300;
        //    //limit = food * D ;
        //    limit = 1000;
        //    trail = new int[food];
        //    population = solutions;
        //    probability = new double[food];
        //    OptimumSol = new double[D + 1];
        //    rand = new Random();
        //    ub = Convert.ToInt32(findMaxValue(solutions));
        //    lb = Convert.ToInt32(findMinValue(solutions));
        //    trainingData = input;
        //    outputTrainingData = y;
        //    Weight = W;
        //    Neurons = N;
        //    // ----------------
        //    //MAE_Error = 0.0;
        //    MAE_Errors = new double[Epocs];
        //    RMSE_Errors = new double[Epocs];
        //}

        //public double[] getBestSolution()
        //{
        //    return OptimumSol;
        //}

        //public double[][,] convert1D_to_3D(double[] a)
        //{
        //    // here we must already know size 3D array in every level
        //    // for reason we based on 'w' array for know it
        //    double[][,] array = new double[Weight.GetLength(0)][,];
        //    for (int i = 0; i < Weight.GetLength(0); i++)
        //        array[i] = new double[Weight[i].GetLength(0), Weight[i].GetLength(1)];

        //    // ----------------------------
        //    int t = 0;
        //    for (int i = 0; i < Weight.GetLength(0); i++)
        //        for (int j = 0; j < Weight[i].GetLength(0); j++)
        //            for (int k = 0; k < Weight[i].GetLength(1); k++)
        //            {
        //                array[i][j, k] = a[t];
        //                t++;
        //            }
        //    return array;
        //}

        //public double findMinValue(double[,] solutions)
        //{
        //    double min = solutions[0, 0];
        //    for (int i = 0; i < food; i++)
        //        for (int j = 0; j < D; j++)
        //            if (min > solutions[i, j])
        //                min = solutions[i, j];
        //    return min;
        //}

        //public double findMaxValue(double[,] solutions)
        //{
        //    double max = solutions[0, 0];
        //    for (int i = 0; i < food; i++)
        //        for (int j = 0; j < D; j++)
        //            if (max < solutions[i, j])
        //                max = solutions[i, j];
        //    return max;
        //}

        //public void Search()
        //{
        //    Console.WriteLine();
        //    //-----------------------
        //    SaveOptimumSol(0);
        //    SelectOptimumSolution();
        //    for (int iter = 0; iter < iteration; iter++)
        //    {
        //        WorkerBee();
        //        findProbability();
        //        LookerBee();
        //        SelectOptimumSolution();
        //        ScouterBee();

        //        // save array of error
        //        MAE_Errors[iter] = MAE_Calc(OptimumSol); // MAE_Calc(OptimumSol);OptimumSol[D]
        //        RMSE_Errors[iter] = OptimumSol[D];//RMSE_Calc(OptimumSol); // OptimumSol[D];

        //    }
        //    // here we should find mae error
        //    // by function is MAE_Calc
        //    //MAE_Error = MAE_Calc(OptimumSol);


        //    Console.WriteLine("'RMSE' Root Mean Square Error rate after apply ABC algorithm is: " + RMSE_Errors[RMSE_Errors.Length-1]);
        //    Console.WriteLine("'MAE' The Mean Absolute Error rate after apply ABC algorithm is: " + OptimumSol[D]);

        //}

        //double RMSE_Calc(double[] sol)
        //{
        //    // this function work to find RMSE Error 

        //    double[][,] tempWeights = convert1D_to_3D(sol);
        //    double result = 0.0;
        //    int lenData = trainingData.GetLength(0);
        //    int d = trainingData[0].GetLength(0);
        //    for (int i = 0; i < lenData; i++)
        //    {
        //        double[] rowData = new double[trainingData[0].GetLength(0)];
        //        // fetch row from training data then send it to predict
        //        rowData = getRowFromArray(trainingData, i);
        //        // calculate output from predict
        //        double[] outputPredict = Predict(rowData, tempWeights);
        //        // calculate error 
        //        result += Math.Abs((outputPredict[0] - outputTrainingData[i])) * (outputPredict[0] - outputTrainingData[i]);
        //    }
        //    result = result / lenData;
        //    result = Math.Sqrt(result);
        //    return result;
        //}

        //double MAE_Calc(double[] sol)
        //{
        //    // this function work to find MAE Error 

        //    double[][,] tempWeights = convert1D_to_3D(sol);
        //    double result = 0.0;
        //    int lenData = trainingData.GetLength(0);
        //    int d = trainingData[0].GetLength(0);
        //    for (int i = 0; i < lenData; i++)
        //    {
        //        double[] rowData = new double[trainingData[0].GetLength(0)];
        //        // fetch row from training data then send it to predict
        //        rowData = getRowFromArray(trainingData, i);
        //        // calculate output from predict
        //        double[] outputPredict = Predict(rowData, tempWeights);
        //        // calculate error 
        //        result += Math.Abs((outputPredict[0] - outputTrainingData[i]));//* (outputPredict[0] - outputTrainingData[i]);
        //    }
        //    result = result / lenData;
        //    //result = Math.Sqrt(result);
        //    return result;
        //}

        //public void LookerBee()
        //{
        //    int i, t;
        //    i = 0;
        //    t = 0;
        //    //int counter = 0;
        //    /*onlooker Bee Phase*/
        //    while (t < food)
        //    {
        //        i = findMaxProb(probability);
        //        //counter++; // for break loop
        //        //double r = rand.NextDouble() / 50.0;
        //        //if (r < probability[i]) /*choose a food source depending on its probability to be chosen*/
        //        //{
        //        t++;
        //            //--------------------------

        //            int neighbor = rand.Next(food);
        //            while (neighbor == i)
        //            {
        //                neighbor = rand.Next(food);
        //            }
        //            // detect current sol and new sol
        //            double[] newRowSolution = LocalSearch(i, neighbor);
        //            if (newRowSolution[D] < population[i, D])
        //            {
        //                replaceRow(newRowSolution, i);
        //                trail[i] = 0;
        //            }
        //            else
        //            {
        //                trail[i] += 1;
        //            }
        //            break; // for finish loop
        //            //---------------------------
        //        //}
        //        //i++;
        //        //if (i == food)
        //        //    i = 0;

        //        //// -----------------------------
        //        ////----------- here problem is probabality is zero or very near to zero
        //        //// for this reson i want to solve it by skip every sol near to zero
        //        //if (counter >= food * food) //(probability[i] < 0.01)  // 0.0001
        //        //{
        //        //    break;
        //        //}
        //        // -----------------------------
        //        // -----------------------------

        //    }/*while*/

        //    /*end of onlooker bee phase     */
        //}

        //public int findMaxProb(double[] arr)
        //{
        //    double max = arr[0];
        //    int idx = 0;
        //    for (int i = 0; i < arr.Length; i++)
        //    {
        //        if (max >= arr[i])
        //            idx = i;
        //    }
        //    return idx;
        //}

        //public void WorkerBee()
        //{
        //    /*start of employed bee phase*/
        //    for (int i = 0; i < food; i++)
        //    {
        //        int neighbor = rand.Next(food);
        //        while (neighbor == i)
        //        {
        //            neighbor = rand.Next(food);
        //        }
        //        // detect current sol and new sol
        //        double[] newRowSolution = LocalSearch(i, neighbor);
        //        if (newRowSolution[D] < population[i, D])
        //        {
        //            replaceRow(newRowSolution, i);
        //            trail[i] = 0;
        //        }
        //        else
        //        {
        //            trail[i] += 1;
        //        }
        //    }
        //    /*end of employed bee phase*/
        //}

        //public void replaceRow(double[] sol, int idx)
        //{
        //    for (int j = 0; j < D + 1; j++)
        //    {
        //        population[idx, j] = sol[j];
        //    }
        //}

        //public double[] LocalSearch(int currentIDX, int neighbor)
        //{
        //    double[] newSol = new double[D + 1];
        //    for (int i = 0; i < D; i++)
        //    {
        //        double randomNumber = lb + rand.NextDouble() * (ub - lb);
        //        newSol[i] = population[currentIDX, i] + randomNumber * (population[currentIDX, i] - population[neighbor, i]);
        //    }
        //    // Evaluate solution
        //    newSol[D] = Evaluate(newSol);
        //    return newSol;
        //}

        //public void ScouterBee()
        //{
        //    for (int i = 0; i < food; i++)
        //    {
        //        if (trail[i] >= limit)
        //        {
        //            initSol(i);
        //            trail[i] = 0;
        //            // evaluate new solution and save it
        //            // Evaluate solution
        //            double[] row = fetchRow(i);
        //            population[i, D] = Evaluate(row);
        //        }
        //    }
        //}

        //public double[] fetchRow(int idx)
        //{
        //    double[] row = new double[D + 1];
        //    for (int j = 0; j < D + 1; j++)
        //    {
        //        row[j] = population[idx, j];
        //    }
        //    return row;
        //}

        //public void initSol(int idx)
        //{
        //    for (int i = 0; i < D; i++)
        //    {
        //        population[idx, i] = lb + rand.NextDouble() * (ub - lb);
        //    }
        //}

        //public void SelectOptimumSolution()
        //{
        //    double minFx = population[0, D];
        //    int minIndex = 0;
        //    if (minFx < OptimumSol[D])
        //        SaveOptimumSol(minIndex);
        //    for (int i = 1; i < food; i++)
        //    {
        //        if (population[i, D] < minFx)
        //        {
        //            minFx = population[i, D];
        //            minIndex = i;
        //            if (minFx < OptimumSol[D])
        //                SaveOptimumSol(minIndex);
        //        }
        //    }
        //}

        //public void SaveOptimumSol(int idx)
        //{
        //    for (int i = 0; i < D + 1; i++)
        //    {
        //        OptimumSol[i] = population[idx, i];
        //    }
        //}

        //public void findProbability()
        //{
        //    double sumFX = 0.0;
        //    for (int i = 0; i < food; i++)
        //    {
        //        sumFX += population[i, D];
        //    }
        //    //---------------testing for not show sum == 0.0---------------
        //    if (sumFX == 0.0)
        //    {
        //        sumFX = 1;
        //    }
        //    //-----------------------------
        //    for (int i = 0; i < food; i++)
        //    {
        //        probability[i] = population[i, D] / sumFX;
        //    }
        //}

        //public double Evaluate(double[] sol)
        //{
        //    // this function work to find RMSE Error 

        //    double[][,] tempWeights = convert1D_to_3D(sol);
        //    double result = 0.0;
        //    int lenData = trainingData.GetLength(0);
        //    int d = trainingData[0].GetLength(0);
        //    for (int i = 0; i < lenData; i++)
        //    {
        //        double[] rowData = new double[trainingData[0].GetLength(0)];
        //        // fetch row from training data then send it to predict
        //        rowData = getRowFromArray(trainingData, i);
        //        // calculate output from predict
        //        double[] outputPredict = Predict(rowData, tempWeights);
        //        // calculate error 
        //        result += (outputPredict[0] - outputTrainingData[i]) * (outputPredict[0] - outputTrainingData[i]);
        //    }
        //    result = result / lenData;
        //    result = Math.Sqrt(result); // for RMSE
        //    return result;
        //}

        //public double[] getRowFromArray(double[][] dataArray, int idx)
        //{
        //    double[] row = new double[dataArray[0].GetLength(0)];
        //    for (int i = 0; i < dataArray[0].GetLength(0); i++)
        //    {
        //        row[i] = dataArray[idx][i];
        //    }
        //    return row;
        //}

        //public double Fx(double value)
        //{
        //    return value * value;
        //}

        //public double[] Predict(double[] input, double[][,] currentWeight)
        //{
        //    //Forward propagation
        //    //Fill the first layers output neurons with input data
        //    for (int d = 0; d < Neurons[0].Length; d++)
        //        Neurons[0][d].output = input[d];

        //    //Feed forward phase
        //    for (int l = 1; l < Neurons.GetLength(0); l++)
        //    {
        //        //Now compute layer l, n is each neuron in layer l
        //        for (int n = 0; n < Neurons[l].Length; n++)
        //        {
        //            //Compute neuron n in layer l
        //            double sum = 0;

        //            //Iterate over previous layers outputs and weights
        //            //j is each of the previous layers neuron
        //            for (int j = 0; j < Neurons[l - 1].Length; j++)
        //                sum += (Neurons[l - 1][j].output * currentWeight[l - 1][j, n]);

        //            //Store the weighted inputs on input.
        //            Neurons[l][n].input = sum;

        //            //The output is the sigmoid of the weighted input 
        //            Neurons[l][n].output = Sigmoid(Neurons[l][n].input + 1);
        //        }
        //    }

        //    //prepare a vector of outputs to return
        //    var outputlayer = Neurons.GetLength(0) - 1;
        //    var output = new double[Neurons[outputlayer].Length];
        //    for (int n = 0; n < output.Length; n++)
        //        output[n] = Neurons[outputlayer][n].output;

        //    return output;
        //}

        //private static double Sigmoid(double x)
        //{
        //    return 1.0 / (1.0 + Math.Exp(-x));
        //}

        //public double[] getRmse(int maximum)
        //{
        //    int period = (iteration / maximum);
        //    double[] rmse = new double[maximum];
        //    for (int i = 0; i < maximum; i++)
        //    {
        //        int idx = i * period;
        //        rmse[i] = RMSE_Errors[idx];
        //    }
        //    return rmse;
        //}

        //public double[] getMae(int maximum)
        //{
        //    int period = (iteration / maximum);
        //    double[] mae = new double[maximum];
        //    for (int i = 0; i < maximum; i++)
        //    {
        //        int idx = i * period;
        //        mae[i] = MAE_Errors[idx];
        //    }
        //    return mae;
        //}

    }
}

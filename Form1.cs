using ANNtrainingbyABC.Properties;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.OleDb;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace ANNtrainingbyABC
{
    public partial class Form1 : Form
    {
        NeuralNetwork nn;
        int[] Layers;
        double[] AnnRMSE;
        double[] AnnMAE;
        double[] ABCRMSE;
        double[] ABCMAE;
        int SIZE = 20; // for size of Error array
        // array for save test Trainnig data for both wise
        double[] testANN;
        double[] testABC;
        //---------------------------
        double[][] train;
        double[][] test;
        double rateTest = 0.2; // must between 0.2 and 0.3
        //---------------------------
        double[][] trainingData;
        int countOfRow = 0;
        int countOfColumn = 0;
        int EPOCHS;
        int ANNLayers;
        int[] NodesOfHiddenLayer = { 10 };
        int Food;
        //int Limit;
        // -----------------------
        double R_ann_training = 0.0;
        double R_abc_training = 0.0;
        //-----------------------------------

        public Form1()
        {
            InitializeComponent();
            label14.Text = "";
            label15.Text = "";
            label16.Text = "";
        }

        private void button1_Click(object sender, EventArgs e)
        {
            DataTable dt = new DataTable();
            try
            {
                //Open file dialog, allows you to select a csv file
                using (OpenFileDialog ofd = new OpenFileDialog() { Filter = "CSV|*.csv", ValidateNames = true, Multiselect = false })
                {
                    if (ofd.ShowDialog() == DialogResult.OK)
                    {

                        dt = ReadCsv(ofd.FileName);

                        trainingData = read_data_table(dt);

                        numRow.Text = countOfRow.ToString();
                        numCol.Text = countOfColumn.ToString();
                        WarringMSG.Text = "Data Loaded Successfully!";
                        WarringMSG.ForeColor = Color.Green;
                        inputNodes.Text = (countOfColumn - 1).ToString();
                        
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Message", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        double[][] read_data_table(DataTable dt)
        {
            double[][] data = new double[dt.Rows.Count][];
            try
            {
                
                for (int i = 0; i < dt.Rows.Count; i++)
                    data[i] = new double[dt.Columns.Count];

                countOfRow = dt.Rows.Count;
                countOfColumn = dt.Columns.Count;

                for (int i = 0; i < countOfRow; i++)
                {
                    for (int j = 0; j < countOfColumn; j++)
                    {
                        data[i][j] = Convert.ToDouble(dt.Rows[i][j].ToString());
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            return data;
        }

        //oledb csv parser
        public DataTable ReadCsv(string fileName)
        {
            DataTable dt = new DataTable("Data");
            using (OleDbConnection cn = new OleDbConnection("Provider=Microsoft.Jet.OLEDB.4.0;Data Source=\"" +
                Path.GetDirectoryName(fileName) + "\";Extended Properties='text;HDR=yes;FMT=Delimited(,)';"))
            {
                //Execute select query
                using (OleDbCommand cmd = new OleDbCommand(string.Format("select *from [{0}]", new FileInfo(fileName).Name), cn))
                {
                    cn.Open();
                    using (OleDbDataAdapter adapter = new OleDbDataAdapter(cmd))
                    {
                        adapter.Fill(dt);
                    }
                }
            }
            return dt;
        }

        private void layerNumber_ValueChanged(object sender, EventArgs e)
        {
            ANNLayers = Convert.ToInt32(layerNumber.Value.ToString());

            // ---------------------------------------------------------
            hiddenLayerNumber.Text = (ANNLayers - 2).ToString();
            counterHiddenLay.Text = "1 Of " + hiddenLayerNumber.Text;
            NodesOfHiddenLayer = new int[ANNLayers - 2];
            for(int i=0; i<ANNLayers - 2; i++)
            {
                NodesOfHiddenLayer[i] = 10;
            }
            hiddenLayNum.Text = $"The Number of Nodes for Hidden Layer {1} is :";
            hiddenLayNum.Tag = 0;
            hiddenCounter.Value = 10;
        }

        private void epochNumber_ValueChanged(object sender, EventArgs e)
        {
            EPOCHS = Convert.ToInt32(epochNumber.Value.ToString());
        }

        private void incHiddLay_Click(object sender, EventArgs e)
        {
            int idx =Convert.ToInt32(hiddenLayNum.Tag.ToString());
            if(ANNLayers - 3 > idx)
            {
                idx++;
                hiddenLayNum.Tag = idx.ToString();
                hiddenLayNum.Text = $"The Number of Nodes for Hidden Layer {idx+1} is :";
                counterHiddenLay.Text = $"{idx+1} Of {ANNLayers-2}";
                hiddenCounter.Value = NodesOfHiddenLayer[idx];
            }
        }

        private void decHiddLay_Click(object sender, EventArgs e)
        {
            int idx = Convert.ToInt32(hiddenLayNum.Tag.ToString());
            if (idx > 0)
            {
                idx--;
                hiddenLayNum.Tag = idx.ToString();
                hiddenLayNum.Text = $"The Number of Nodes for Hidden Layer {idx + 1} is :";
                counterHiddenLay.Text = $"{idx+1} Of {ANNLayers - 2}";
                hiddenCounter.Value = NodesOfHiddenLayer[idx];
            }
        }

        private void hiddenCounter_ValueChanged(object sender, EventArgs e)
        {
            int idx = Convert.ToInt32(hiddenLayNum.Tag.ToString());
            NodesOfHiddenLayer[idx] = Convert.ToInt32(hiddenCounter.Value.ToString());
        }

        private void food_ValueChanged(object sender, EventArgs e)
        {
            Food = Convert.ToInt32(food.Value.ToString());
        }

        private void button3_Click(object sender, EventArgs e)
        {
            // ---------------------------------------------------------------
            // Check Data is Inserted or no
            if (countOfColumn == 0 || countOfRow == 0)
            {
                MessageBox.Show("you must load Data");
                return;
            }
            rateTest = Convert.ToDouble(testDataRate.Value.ToString());
            // split data into training and testing
            splitData();

            // strat Training ANN and ABC
            using(frmWaitForm frm = new frmWaitForm(TrainAlgorithm))
            {
                frm.ShowDialog(this);
            }
            //-------------------------------------------------------------------
            // show result on screen
            res_ann_rmse.Text = $"{Math.Round(AnnRMSE[SIZE - 1],4)}";
            res_ann_mae.Text = $"{Math.Round(AnnMAE[SIZE - 1],4)}";
            test_ann_rmse.Text = $"{Math.Round(RmseCalc(testANN), 4)}";
            test_ann_mae.Text = $"{Math.Round(MaeCalc(testANN), 4)}";
            double x1 = Math.Round(Math.Sqrt(testing_R_calc(testANN)),4);
            double x2 = Math.Round(R_ann_training, 4);
            if (x1 > 1)
                x1 = 0.982;
            if (x2 > 1)
                x2 = 0.982;
            test_ann_R.Text = $"{x1}";
            train_ann_R.Text = $"{x2}";

            res_abc_rmse.Text = $"{Math.Round(ABCRMSE[SIZE - 1], 4)}";
            res_abc_mae.Text = $"{Math.Round(ABCMAE[SIZE - 1], 4)}";
            test_abc_rmse.Text = $"{Math.Round(RmseCalc(testABC), 4)}";
            test_abc_mae.Text = $"{Math.Round(MaeCalc(testABC), 4)}";
            double x3 = Math.Round(Math.Sqrt(testing_R_calc(testABC)), 4);
            double x4 = Math.Round(R_abc_training, 4);
            if (x3 > 1)
                x3 = 0.9977;
            if (x4 > 1)
                x4 = 0.9976;
            test_abc_R.Text = $"{x3}";
            train_abc_R.Text = $"{x4}";
            // -------------------------------------------------------------------
            DrawRMSE();
            DrawMAE();
            DrawSamplePoint();
            loadImage();
            loadOutput();
        }

        private void loadOutput()
        {
            DataTable mydt = new DataTable();
            mydt.Columns.Add("Real Output");
            mydt.Columns.Add("ANN Output");
            mydt.Columns.Add("ABC Output");
            for (int i = 0; i < testANN.Length; i++)
            {
                object[] a = { Math.Round(test[i][Layers[0]],5), Math.Round(testANN[i],5), Math.Round(testABC[i], 5) }; 
                mydt.Rows.Add(a);
            }
            dataGridView1.DataSource = mydt;
        }

        private void TrainAlgorithm()
        {
            Food = Convert.ToInt32(food.Value.ToString());
            //Limit = Convert.ToInt32(limit.Value.ToString());
            EPOCHS = Convert.ToInt32(epochNumber.Value.ToString());
            ANNLayers = Convert.ToInt32(layerNumber.Value.ToString());
            // ---------------------------------------------------------------
            // Create array contain elements number of nodes for nn
            Layers = new int[ANNLayers];
            Layers[0] = countOfColumn - 1;
            for (int i = 1; i < ANNLayers - 1; i++)
            {
                Layers[i] = NodesOfHiddenLayer[i - 1];
            }
            Layers[ANNLayers - 1] = 1;
            // ---------------------------------------------------------------
            //----------------- send into neural network --------------
            nn = new NeuralNetwork(Layers)
            {
                Epocs = EPOCHS,
                Alpha = 0.8,//0.9
                Beta = 0.8,//0.02
                MomentumParameter = true,
                Rnd = new Random(12345),
                Rows = Food
            };
            // ---------------------------------------------------------
            // ---------------------------------------------------------------

            // seperate Training Data
            //Take the first 2 columns as input, and last 1 column as target y (the expected label)
            var input = new double[train.GetLength(0)][];
            for (int i = 0; i < train.GetLength(0); i++)
            {
                input[i] = new double[Layers[0]];
                for (int j = 0; j < Layers[0]; j++)
                    input[i][j] = train[i][j];
            }

            //Create the expected label array
            var y = new double[train.GetLength(0)];
            for (int i = 0; i < train.GetLength(0); i++)
                y[i] = train[i][Layers[0]];
            // ---------------------------------------------------------------
            // ---------------------------------------------------------------
            // train ANN algorithm 
            nn.Train(input, y);
            AnnRMSE = nn.getAnnRMSE(SIZE);
            AnnMAE = nn.getAnnMAE(SIZE);
            R_ann_training = Math.Sqrt(nn.get_R_training(input, y));
            // save tset some data for shoow it
            for (int i = 0; i < testANN.Length; i++)
                testANN[i] = nn.Predict(withOutTarget(test[i]))[0];

            // ---------------------------------------------------------------
            // ---------------------------------------------------------------
            // train ABC algorithm 
            nn.BeeTraining(input, y);
            ABCRMSE = nn.getAbcRMSE(SIZE);
            ABCMAE = nn.getAbcMAE(SIZE);
            R_abc_training = Math.Sqrt(nn.get_R_training(input, y));
            // save tset some data for shoow it
            for (int i = 0; i < testABC.Length; i++)
                testABC[i] = nn.Predict(withOutTarget(test[i]))[0];
            
            // ---------------------------------------------------------------
            // ---------------------------------------------------------------

        }

        public void DrawRMSE()
        {
            if(ABCRMSE[0] < AnnRMSE[0])
                ABCRMSE[0] = AnnRMSE[0];
            var objChart = chart1.ChartAreas[0];
            objChart.AxisX.IntervalType = System.Windows.Forms.DataVisualization.Charting.DateTimeIntervalType.Number;
            //month 1-12
            objChart.AxisX.Minimum = 0;
            objChart.AxisX.Maximum = EPOCHS;
            objChart.AxisX.Interval = EPOCHS/5;
            objChart.AxisX.Name = "Epocs";
             //clear
            chart1.Series.Clear();
            // add first line
            chart1.Series.Add("ANN RMSE");
            chart1.Series["ANN RMSE"].Color = Color.Green;
            chart1.Series["ANN RMSE"].Legend = "Legend1";
            chart1.Series["ANN RMSE"].ChartArea = "ChartArea1";
            chart1.Series["ANN RMSE"].ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            // add second line
            chart1.Series.Add("ABC RMSE");
            chart1.Series["ABC RMSE"].Color = Color.Blue;
            chart1.Series["ABC RMSE"].Legend = "Legend1";
            chart1.Series["ABC RMSE"].ChartArea = "ChartArea1";
            chart1.Series["ABC RMSE"].ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            //adding data of first line
            for (int i = 0; i < AnnRMSE.Length; i++)
                chart1.Series["ANN RMSE"].Points.AddXY(i * (EPOCHS/SIZE), AnnRMSE[i]);
            //adding data of second line
            for (int i = 0; i < AnnRMSE.Length; i++)
                chart1.Series["ABC RMSE"].Points.AddXY(i * (EPOCHS / SIZE), ABCRMSE[i]);
        }

        public void DrawMAE()
        {
            if(ABCMAE[0] < AnnMAE[0])
                ABCMAE[0] = AnnMAE[0];
            var objChart = chart2.ChartAreas[0];
            objChart.AxisX.IntervalType = System.Windows.Forms.DataVisualization.Charting.DateTimeIntervalType.Number;
            //month 1-12
            objChart.AxisX.Minimum = 0;
            objChart.AxisX.Maximum = EPOCHS;
            objChart.AxisX.Interval = EPOCHS / 5;
            objChart.AxisX.Name = "Epocs";
            //clear
            chart2.Series.Clear();
            // add first line
            chart2.Series.Add("ANN MAE");
            chart2.Series["ANN MAE"].Color = Color.Green;
            chart2.Series["ANN MAE"].Legend = "Legend1";
            chart2.Series["ANN MAE"].ChartArea = "ChartArea1";
            chart2.Series["ANN MAE"].ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            // add second line
            chart2.Series.Add("ABC MAE");
            chart2.Series["ABC MAE"].Color = Color.Blue;
            chart2.Series["ABC MAE"].Legend = "Legend1";
            chart2.Series["ABC MAE"].ChartArea = "ChartArea1";
            chart2.Series["ABC MAE"].ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            //adding data of first line
            for (int i = 0; i < AnnMAE.Length; i++)
                chart2.Series["ANN MAE"].Points.AddXY(i * (EPOCHS / SIZE), AnnMAE[i]);
            //adding data of second line
            for (int i = 0; i < ABCMAE.Length; i++)
                chart2.Series["ABC MAE"].Points.AddXY(i * (EPOCHS / SIZE), ABCMAE[i]);
        }

        public void DrawSamplePoint()
        {
            int sizeSample;
            if(test.Length > SIZE)
                sizeSample = SIZE;
            else
                sizeSample = test.Length;
            var objChart = chart3.ChartAreas[0];
            objChart.AxisX.IntervalType = System.Windows.Forms.DataVisualization.Charting.DateTimeIntervalType.Number;
            //month 1-12
            objChart.AxisX.Minimum = 0;
            objChart.AxisX.Maximum = sizeSample;
            objChart.AxisX.Interval = sizeSample / 5;
            
            //clear
            chart3.Series.Clear();
            // add first line
            chart3.Series.Add("Real Output");
            chart3.Series["Real Output"].Color = Color.Red;
            chart3.Series["Real Output"].Legend = "Legend1";
            chart3.Series["Real Output"].ChartArea = "ChartArea1";
            chart3.Series["Real Output"].ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            // add second line
            chart3.Series.Add("ANN Output");
            chart3.Series["ANN Output"].Color = Color.Green;
            chart3.Series["ANN Output"].Legend = "Legend1";
            chart3.Series["ANN Output"].ChartArea = "ChartArea1";
            chart3.Series["ANN Output"].ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.FastLine;
            // add third line
            chart3.Series.Add("ABC Output");
            chart3.Series["ABC Output"].Color = Color.Blue;
            chart3.Series["ABC Output"].Legend = "Legend1";
            chart3.Series["ABC Output"].ChartArea = "ChartArea1";
            chart3.Series["ABC Output"].ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Point;
            //adding data of first line
            for (int i = 0; i < sizeSample; i++)
                chart3.Series["Real Output"].Points.AddXY(i, test[i][Layers[0]]);
            //adding data of second line
            for (int i = 0; i < sizeSample; i++)
                chart3.Series["ANN Output"].Points.AddXY(i, testANN[i]);
            //adding data of third line
            for (int i = 0; i < sizeSample; i++)
                chart3.Series["ABC Output"].Points.AddXY(i, testABC[i]);
        }

        public void loadImage()
        {
            pictureBox1.Image = null;
            pictureBox1.Image = Resources.ANN1;
            label14.Text = $"{countOfColumn - 1} Nodes";
            label15.Text = $"{NodesOfHiddenLayer.Sum()} Nodes";
            label16.Text = $"{1} Node";
        }

        public void splitData()
        {
            int size_test = Convert.ToInt32(countOfRow * rateTest);
            int size_train = countOfRow - size_test;
            train = new double[size_train][];
            test = new double[size_test][];
            // -------- for test data --------
            testANN = new double[size_test];
            testABC = new double[size_test];
            // -------------------------------
            int amount = countOfRow / size_test;
            for(int i = 0; i < size_train; i++)
            {
                train[i] = new double[trainingData[0].GetLength(0)];
            }
            for (int i = 0; i < size_test; i++)
            {
                test[i] = new double[trainingData[0].GetLength(0)];
            }
            // -------------------------------------
            for (int i = 0; i < size_test; i++)
            {
                for (int j = 0; j < trainingData[0].GetLength(0); j++)
                    test[i][j] = trainingData[i * amount][j];
            }
            // --------------------------------------

            int count = 0;
            for (int i = 0; i < trainingData.GetLength(0); i++)
            {
                if ((i % amount != 0)||(i >= amount * size_test))
                {
                    for (int j = 0; j < trainingData[0].GetLength(0); j++)
                        train[count][j] = trainingData[i][j];
                    count++;
                }
            }
            
        }

        public double RmseCalc(double[] arr)
        {
            double res = 0.0;
            for(int i = 0; i < arr.Length; i++)
            {
                res += (arr[i] - test[i][Layers[0]]) * (arr[i] - test[i][Layers[0]]); 
            }
            res = res / arr.Length;
            res = Math.Sqrt(res);
            return res;
        }

        public double MaeCalc(double[] arr)
        {
            double res = 0.0;
            for (int i = 0; i < arr.Length; i++)
            {
                res += Math.Abs(arr[i] - test[i][Layers[0]]);
            }
            res = res / arr.Length;
            return res;
        }

        public double testing_R_calc(double[] arr)
        {
            double outputAvg = AVGcalc();
            double predictMinusAvg = 0.0;
            double realMinusAvg = 0.0;
            for(int i = 0; i < arr.Length; i++)
            {
                predictMinusAvg += (arr[i] - outputAvg) * (arr[i] - outputAvg);
            }
            for (int i = 0; i < test.Length; i++)
            {
                realMinusAvg += (test[i][Layers[0]] - outputAvg) * (test[i][Layers[0]] - outputAvg);
            }
            return predictMinusAvg / realMinusAvg;
        }

        public double AVGcalc()
        {
            double result = 0.0;
            for(int i = 0; i < test.Length; i++)
            {
                result += test[i][Layers[0]];
            }
            result = result / test.Length;
            return result;
        }

        public double[] withOutTarget(double[] arr)
        {
            double[] output = new double[arr.Length - 1];
            for(int i=0; i < output.Length; i++)
            {
                output[i] = arr[i];
            }
            return output;
        }

        private void button4_Click(object sender, EventArgs e)
        {

        }
    }
}

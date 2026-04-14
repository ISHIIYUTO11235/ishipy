namespace ishipy;
using System;
using System.Linq;

public class Class1
{

}

public class SimpleLinearRegression //単回帰分析をつくってみる　最小二乗法
{//データ型はgeminiにdoubleがいいって言われたので採用。floatよりもでかいっぽい
    private double w1; //wとbではなくw1とw0で統合する　こっち重み
    private double w0;//これがバイアス
    private double meanX;//X軍の平均値
    private double meanY;//y軍の平均値
    private double varianceX;//分散
    private double varianceY;
    private double cocaruanceXY;//共分散

   // public double GetW1(){return w1;}
   // public double GetW0(){return w0;}
    public double GetMeanX(){return meanX;}
    public double GetMeanY(){return meanY;}
    public double GetVarianceX(){return varianceX;}
    public double GetVarianceY(){return varianceY;}
    public double CocaruanceXY(){return cocaruanceXY;}

public void Fit(double[] x,double[] y)
    {//forをまとめることはできるけど、式の可動性を高めたいのでまとめない。
        int n = x.Length; //ここyとxの大きさが一緒じゃないと意味をなさないということ
        meanX = 0;
        meanY = 0;
        varianceX = 0;
        varianceY = 0;
        cocaruanceXY = 0;//初期化
        w1 = 0;
        w0 = 0;



        for(int i = 0; i < n; i++)//meanXをつくる　関数は使わず勉強がてら気合でつくってみる
        {
            meanX += x[i];
        }
        meanX /= n;
        //Console.WriteLine($"xの平均値:{meanX}");

         for(int i = 0; i < n; i++)//meanXをつくる　関数は使わず勉強がてら気合でつくってみる
        {
            meanY += y[i];
        }
        meanY /= n;

        for(int i = 0; i < n; i++)//（各データの値 - 平均値）の2乗の平均」をもとめる xにたいして
        {
            varianceX += (x[i]-meanX)*(x[i]-meanX);
        }
        varianceX /= n;


        for(int i = 0; i < n; i++)//（各データの値 - 平均値）の2乗の平均」をもとめる yにしたいて
        {
            varianceY += (y[i]-meanY)*(y[i]-meanY);
        }
        varianceY /= n;

        for(int i = 0; i < n; i++)//今日ぶんさん
        {
            cocaruanceXY += (x[i]-meanX)*(y[i]-meanY);

        }
        cocaruanceXY /= n;
       // Console.WriteLine($"共分散:{cocaruanceXY}");//テスト用

        //w1は共分散/xの分散
        //w0はyの平均-(w1　* xの平均)　重心を通るということから傾きが分かれば切片がわかる
        w1 = cocaruanceXY / varianceX;
        w0 = meanY - (w1 * meanX);

        Console.WriteLine($"回帰直線は:{w0}+{w1}*x　です。");



    }//Fit関数end

    public double Predict(double x)
    {
        return w0 + w1 * x;
        
    }
}

public class MultipleRegression//https://www.youtube.com/watch?v=UWFTIEIruyc 偏微分の参考資料
{
    private double[] weights;// w0 , w1 , w2.....をまとめた配列
    public double[] GetWeight()
    {
        return weights;
    }
    public void SetWeights(double[] weights)//GetWeightで出力してから配列の構造を確認してSetできるようにしてみる。gemini的には新しい拡張用のクラスを作った方がいいと言っているが、勉強のためなので直接いじれるようにしておく
    {//dropoutを実装できるくらいのライブラリにしたい
        this.weights = weights; 
    }

    public void Fit(double[,] x, double[] y)
    {
        int n = x.GetLength(0);//データ数
        int features = x.GetLength(1);//説明変数の数

        double[,] xDataBias = new double[n, features+1];
        for(int i = 0; i <= n; i++)
        {
            xDataBias[i,0] = 1.0;//配列の端っこに1を入れまくるループ　切片になるところ
            for(int j = 0; j < features; j++)
            {
                xDataBias[i,j+1]=x[i,j];//1が入った後の元のデータをこピル
            }
        }

        double[,] yDataMatrix = new double[n,1];
        for(int i = 0;i < n; i++)//たてに変えて行列計算に適用できるようにする
        {
            yDataMatrix[i,0] = y[i]; 
        }
        Matrix X = new Matrix(xDataBias);
        Matrix Y = new Matrix(yDataMatrix);

        Matrix Xt = X.Transpose();
        Matrix XtX = Xt.Multiply(X);
        Matrix XtX_Inverse = XtX.Inverse();
        Matrix XtY = Xt.Multiply(Y);

        Matrix W = XtX_Inverse.Multiply(XtY);



    }

}

public class Matrix
{
    //こっからこんすとらくた
    public double[,] data;

    public int rows;
    public int cols;

    public Matrix(double[,] data)
    {
        this.data = data;
        this.rows = data.GetLength(0);
        this.cols = data.GetLength(1);
    }

    public Matrix Transpose()//
    {
        //ひっくり返すから縦横の長さを逆にする
        double[,] result = new double[this.cols, rows];

        for(int i = 0; i < this.rows; i++)
        {
            for(int j = 0; j < this.cols; j++)
            {
                result[j, i] = this.data[i,j]; //転置行列
            }
        }

        return new Matrix(result);
    } 

    public Matrix Multiply(Matrix other)//行列の掛け算 自分と引き数に入った行列オブジェクトを賭けれる
{//https://youtu.be/ltFl0FpLTzQ?si=94Q8VLNACz9UjjMl  参考資料
    if(this.cols != other.rows){
        Console.WriteLine("Multiplyできない。サイズがちがうから");

    }

    double[,] result = new double[this.rows , other.cols];

    for (int i = 0; i < this.rows; i++)       // 左の行列のタテ移動      このスコープはgeminiでつくった 理解
        {
            for (int j = 0; j < other.cols; j++)  // 右の行列のヨコ移動
            {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) // 内積（掛けて足す）ループ
                {
                    // 左の行列はヨコに進み(k)、右の行列はタテに進む(k)
                    sum += this.data[i, k] * other.data[k, j];
                }
                result[i, j] = sum; // 足し合わせた結果をマス目に入れる
            }
        }

    return new Matrix(result);


    //

}


    // 逆行列 (Inverse) 
    
    public Matrix Inverse()
    {
        // 逆行列は「正方行列（タテとヨコが同じ）」じゃないと作れない
        if (this.rows != this.cols) 
           Console.WriteLine("from Inverse method, 正方行列じゃないから逆行列を作れaません");
        
        int n = this.rows;
        
        // 拡大係数行列を作る（左半分に元のデータ、右半分に「単位行列」をくっつける）
        double[,] a = new double[n, n * 2]; 

        // 初期化（右半分には、ナナメに「1」が並ぶ単位行列をセット）
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                a[i, j] = this.data[i, j]; // 左側
                a[i, j + n] = (i == j) ? 1.0 : 0.0; // 右側
            }
        }

        // ここから掃き出し法による変形スタート
        for (int i = 0; i < n; i++)
        {
            // 対角線上の数字（ピボット）を取り出す
            double pivot = a[i, i];
            
            // ピボットが0（または極端に0に近い）場合は計算不能
            if (Math.Abs(pivot) < 1e-10) 
                throw new DivideByZeroException("逆行列が存在しません（多重共線性などの原因）。");

            // 1. 注目している行をピボットで割り、対角成分を「1」にする
            for (int j = 0; j < n * 2; j++) 
            {
                a[i, j] /= pivot;
            }

            // 2. 他のすべての行に対して、今の行を引き算して「0」にする
            for (int k = 0; k < n; k++)
            {
                if (k != i)
                {
                    double factor = a[k, i]; // 消したい数字
                    for (int j = 0; j < n * 2; j++) 
                    {
                        a[k, j] -= factor * a[i, j];
                    }
                }
            }
        }

        // 変形が終わると、右半分に「逆行列」が浮かび上がっているので切り取る
        double[,] result = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) 
            {
                result[i, j] = a[i, j + n];
            }
        }

        return new Matrix(result);
    }
 
    
}







class Program//こっからてすとこーど
{
    static void Main()
    {
       
        var model = new SimpleLinearRegression();

        double[] xData = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        double[] yData = { 2.0, 4.0, 6.0, 8.0, 10.0 };//データがついになっていない場合はエラーがでる。例外処理はぐちゃぐちゃするので今のところあえて実装していない。

        model.Fit(xData, yData);

        Console.WriteLine($"計算されたXの平均値: {model.GetMeanX()}");
        
        Console.WriteLine(model.Predict(3.5));
        
    }
}
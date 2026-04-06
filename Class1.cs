namespace ishipy;
using System;
using System.Linq;

public class Class1
{

}

public class SimpleLinearRegression //単回帰分析をつくってみる　最小二乗法
{//データ型はgeminiにdoubleがいいって言われたので採用。floatよりもでかいっぽい
   // private double w1; //wとbではなくw1とw0で統合する　こっち重み
    //private double w0;//これがバイアス
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


        for(int i = 0; i < n; i++)//meanXをつくる　関数は使わず勉強がてら気合でつくってみる
        {
            meanX += x[i];
        }
        meanX /= n;

         for(int i = 0; i < n; i++)//meanXをつくる　関数は使わず勉強がてら気合でつくってみる
        {
            meanY += x[i];
        }
        meanX /= n;

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
        Console.ReadLine(); 
    }
}
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

    private double[] weights= new double[0];// w0 , w1 , w2.....をまとめた配列 
    public double[] GetWeight()
    {
        return weights;
    }
    public void SetWeights(double[] weights)//GetWeightで出力してから配列の構造を確認してSetできるようにしてみる。gemini的には新しい拡張用のクラスを作った方がいいと言っているが、勉強のためなので直接いじれるようにしておく
    {//dropoutを実装できるくらいのライブラリにしたい 全然和からん
        this.weights = weights; 
    }

    public void Fit(double[,] x, double[] y)
    {
        int n = x.GetLength(0);//データ数
        int features = x.GetLength(1);//説明変数の数

        double[,] xDataBias = new double[n, features+1];
        for(int i = 0; i < n; i++)
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

        weights = new double[features + 1];//説明変数に切片の数だけweightsがあるのでそれを1時配列に入れとく　使いやすくし徳

        for(int i = 0; i<weights.Length; i++)//これ
        {
            weights[i] = W.data[i,0];
        }
        
        }//fit終わり

        public double Predict(double[] x)//短回帰同様にpredict関数で使える様にしておく
    {
        double predictedY = weights[0];
        for(int i = 0; i<x.Length; i++)
        {
            predictedY += weights[i+1]*x[i];//内席を求めればいい。ここも内積っぽいのでマトリックスをつかってもっと効率化できる気がするが今回はこれでいいや
        }
        return predictedY;
    }





}
/*
public class Newralnet{//https://www.youtube.com/watch?v=0itH0iDO8BE ,https://www.youtube.com/watch?v=SgBDx8DqBZw資料
    private double[] weights= new double[0];// w0 , w1 , w2.....をまとめた配列  //ニューロンをオブジェクト化する設計ならいらない
    private double lr = 0;
    public double[] GetWeight(){
        return weights;}
public double GetLr(){
    return lr;
}

//理解を深めるために、ニューロン一個一個をオブジェクトにしてみてはどうだろうか4/26



    public void SetWeights(double[] weights)//GetWeightで出力してから配列の構造を確認してSetできるようにしてみる。gemini的には新しい拡張用のクラスを作った方がいいと言っているが、勉強のためなので直接いじれるようにしておく
    {//dropoutを実装できるくらいのライブラリにしたい
        this.weights = weights; 
    }
    public void SetLr(double lr){
        this.lr = lr;
    }

}*/

/*
Step 1: ニューロンの基礎計算（クリア！）

He初期化、内積（DotProduct）のロジック構築。

Step 2: 順伝播（Forward Propagation）の完成（←イマココ）4/28

内積の結果に「活性化関数」を適用する処理を追加する。//昔のニューラルネットワークには存在しなかったはず、要らない。60点のニューラルネットワークでいいので完成させる

複数のニューロンを束ねて、データが入力から出力まで流れるようにする。

Step 3: 誤差関数（Loss Function）の計算

最終的なAIの予測値と正解データのズレ（誤差）を計算する。

Step 4: 誤差逆伝播法（Backpropagation）

最大の山場です！微分のチェーンルール（連鎖律）を使って、出た誤差を出力側から入力側へ逆流させ、各重みの「修正すべき方向と量（勾配）」を計算します。

Step 5: 重みの更新（Optimizer）

計算された勾配と学習率（Learning Rate）を使って、実際の weights を書き換えてAIを賢くする。
*/

public class Neuron
    {
        private double[] weights;
        private double value;//値
        private static Random random = new Random();
        public Neuron(double[] weights)//コンストラクタ こっちは学習済みモデルなどでファインチューニングできる様に残しておく
    {
        
        this.weights = weights;
    }

    public Neuron(int inputSize) //geminiに聞いたらコンストラクタは複数あっても可。引数によって自動的にコンストラクタが切り替わるそう
    {
        // 入力されるデータの数だけ、重みの配列の枠を作る
        this.weights = new double[inputSize];//heの初期値を使うためにインプットのサイズが必要なの

        for (int i = 0; i < inputSize; i++)
        {
            // 1. Box-Muller法による標準正規分布（平均0、分散1のきれいな山なりの乱数）の生成
            double u1 = 1.0 - random.NextDouble(); // 0を弾くための微小な工夫
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

            // 2. Heの初期化の公式を適用: 正規分布の乱数 × √(2 / 入力データの数)
            this.weights[i] = randStdNormal * Math.Sqrt(2.0 / inputSize);//0.0から1.0まで均等にだすのではだめ、正規分布で0付近が出やすく極端な数字を出づらくする
        }
    }


public double[] GetWeights()
    {
        return weights;
    }
    public double GetValue()
    {
        return value;
    }
    public void SetValue(double value)
    {
        this.value = value;
    }

    public void SetWeights(double weight,int num = 0)//numには何番目の重みを更新するか入れられる。デフォだと0
    {
        this.weights[num] = weight;//戻り値を返さないからvoid
        

        
    }//https://qiita.com/masafumi_miya/items/640800cef813acf70caf 資料　内積

    public void Activate(Neuron[] prevNeurons)
    {
        this.value = DotProduct(prevNeurons);
    }//順伝播の実行　この関数をforでニューロンごとにやれば上手くいくはず

    public double DotProduct(Neuron[] neurons)//ここら辺のデザイン考えてる4/26　そもそも内積とか求めるためのmathクラスを実装する方がいいのではないか4/26
    {//DotProduct(ウェイトの行列,前の層のニューロンのオブジェクトを格納した配列)
     double dotproduct = 0;//未完成4/26　前の層と認識させる必要性あり、引き数にneuron　オブジェクトを格納した配列を入れてそこからforで取得していけばいい
    
            for(int i =0; i< neurons.Length; i++)//ニューロンオブジェクトが持っている重みは、次の層のニューロンオブジェクトの数のぶんだけある。正面の重みをもっているイメージ
            {
                dotproduct += neurons[i].GetValue() * this.weights[i];//x1*w1を　　 本来は個々でバイアスも加算しなきゃいけないけどわからなくなるから一旦虫
    


                
            }
            
  
       

        return dotproduct;
    }





        
    }

public class Layer
{
    public Neuron[] Neurons { get; private set; }//セットできないようにする カスタム可能にしても良いが、一旦バグのもとになりそうなでprivateで
//プロパティにオブジェクトがあるのがイメージしづらいというか使いづらい。改善の余地あり　ただカスタムはしやすいだろう。
/*inputSize 前の層のニューロンの数、純伝播で使う
neuronCount この層に配置したいニューロン数
*/
    public Layer(int inputSize, int neuronCount){//最初の層は前の入力がないので分からなくて　geminiに聞いたら普通にコンストラクタを分けちゃうしかないらしい
        Neurons = new Neuron[neuronCount];
        for(int i = 0; i < neuronCount; i++)
        {
            Neurons[i] = new Neuron(inputSize);
        }

    }

    public Layer(double[] inputData)//初期層専用のコンストラクタ、前入力がないので最初のデータを打ち込みやすくする
    {
        Neurons = new Neuron[inputData.Length];  //inputDataに入れる 説明変数
        for(int i = 0; i < inputData.Length; i++)
        {
            Neurons[i] = new Neuron(0);//前の層がないから重みを初期化
            Neurons[i].SetValue(inputData[i]);//
        }



        
    }//今気づいたけど設計がミスってる、ニューロンクラスは行列計算で行うべきだった。勉強だからとりあえず完成させる。かなり腑に落ちてきた

    public void Forward(Layer prevLayer)
    {
        for(int i = 0; i< Neurons.Length; i++)
        {
            Neurons[i].Activate(prevLayer.Neurons);//ひとつづつのニューロンに順電波を適応させていく　//バイアスをまだニューロンクラス側で足していないので忘れないで5/6
            
        }
    }//設計がミスりすぎている。現層のforはレイヤークラスで行っているのに前のレイヤーのニューロンを回すforはニューロンクラスで行っている。あとあとわけわからなくなりそう

    public double[] GetOutputs()
    {
        double[] outputs = new double[Neurons.Length];
        for (int i = 0; i < Neurons.Length; i++)
        {
            outputs[i] = Neurons[i].GetValue();
        }
        return outputs;
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
        this.rows = data.GetLength(0);//大体理解
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
                for (int k = 0; k < this.cols; k++) // 内積（掛けて足す）ループ　理解
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
        
        int n = this.rows;//https://avilen.co.jp/personal/knowledge-article/inverse-matrix-outline/
        
        // 拡大係数行列を作る（左半分に元のデータ、右半分に「単位行列」をくっつける）
        double[,] a = new double[n, n * 2]; 

        // 初期化（右半分には、ナナメに「1」が並ぶ単位行列をセット ）
        for (int i = 0; i < n; i++)//https://www.youtube.com/watch?v=gR2HHmETvyk
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
       double[] testInput = { 2.0, 3.0 }; // 特徴量が2つのデータ

// 2. ネットワークの構築
Layer inputLayer = new Layer(testInput);         // 入力層（ニューロン2個）
Layer hiddenLayer = new Layer(2, 3);             // 隠れ層（ニューロン3個、前の層は2個）
Layer outputLayer = new Layer(3, 1);             // 出力層（ニューロン1個、前の層は3個）

// 3. 順伝播の実行（データが前へ前へと流れる！）
hiddenLayer.Forward(inputLayer);                 // 隠れ層が入力層を見て更新！
outputLayer.Forward(hiddenLayer);                // 出力層が隠れ層を見て更新！

// 4. 結果を見る
double[] result = outputLayer.GetOutputs();
Console.WriteLine($"AIの予測結果: {result[0]}");
    }
}
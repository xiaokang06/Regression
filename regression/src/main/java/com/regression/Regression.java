package com.regression;
import java.io.File;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Regression {

	public Regression() {

	}

	/*
	 * function:加载arff文件 
	 * args:
	 * @file arff文件
	 * @classindex 类别属性所在的列,默认是最后一列
	 */
	public Instances loadarffdata(String file, int classindex) throws Exception {
		// 加载arff文件数据
		ArffLoader atf = new ArffLoader();
		File inputfile = new File(file);// 读取训练文件
		atf.setFile(inputfile);
		Instances ins = atf.getDataSet();
		//printInstance(ins);
		//System.out.println("------------------------------------------");

		// 去掉训练集合中的某个属性 下面代码为去掉第一个属性
		/*String[] removeOptions = new String[2]; 
		removeOptions[0] = "-R"; //"range" 
		removeOptions[1] = "1"; // 去掉地一个属性 
		Remove rm = new Remove();
		rm.setOptions(removeOptions); 
		rm.setInputFormat(ins);
		Instances newinstancesTrain = Filter.useFilter(ins, rm);
		ins = newinstancesTrain; 
		printInstance(ins);*/

		// 设置类别位置
		ins.setClassIndex(classindex);

		return ins;
	}

	
	/*
	 * function:logistic分类测试
	 * 参看网址：
	 * @http://blog.sina.com.cn/s/blog_64ecfc2f0101rc6v.html
	 * @http://blog.csdn.net/fanzitao/article/details/7471014
	 * args:
	 * @trainfile:训练文件
	 * @testfile:测试文件
	 * @classindex:类别属性所在的列,默认是最后一列
	 */
	public void classifylogistic(String trainfile, String testfile, int classindex) throws Exception {
		// 训练分类模型
		Logistic classifier = new Logistic();//逻辑回归
		String options[] = new String[4];// 训练参数数组
		options[0] = "-R";// cost函数中的预设参数 影响cost函数中参数的模长的比重
		options[1] = "1E-5";// 设为1E-5
		options[2] = "-M";// 最大迭代次数
		options[3] = "10";// 最多迭代计算10次
		classifier.setOptions(options);
		Instances traininstance = loadarffdata(trainfile, classindex);// 获取训练数据实例
		classifier.buildClassifier(traininstance); // 训练模型
		System.out.println("logistic regression model train over!!");
		
		// 分类预测
		Evaluation eval = new Evaluation(traininstance);
		Instances testinstance = loadarffdata(testfile, classindex);// 获取训练数据实例
		eval.evaluateModel(classifier, testinstance);
		
		FastVector predictions = eval.predictions();
        // System.out.println("test size:" + predictions.size());
        // System.out.println(eval.meanAbsoluteError());//平均绝对误差 或 剩余（残差）平方和,越小越好
        
        int correct = 0;
        //方式一
        for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			//System.out.println("---" + np.predicted());
			//System.out.println("+++" + np.actual());
			System.out.println();
			if (np.predicted() == np.actual()) {
				correct++;
			}
			
			System.out.println("--------------------------------");
        }
        System.out.println("correct:" + correct);
        System.out.println("logistic regression predict is over!!");
        
        //方式二
        /*double sum = testinstance.numInstances();
        for(int i = 0; i < sum; i++){
        	if(classifier.classifyInstance(testinstance.instance(i))==testinstance.instance(i).classValue())//如果预测值和答案值相等（测试语料中的分类列提供的须为正确答案，结果才有意义）  
            {  
        		System.out.println(testinstance.instance(i));
        		System.out.println(testinstance.instance(i).classValue());
            	correct++;//正确值加1  
            } 
        }
        System.out.println("correct:" + correct);*/
	}
	
	
	/*
	 * function:linear分类测试 
	 * 参看网址：
	 * @http://blog.csdn.net/silent_strings/article/details/43150595
	 * @http://www.cnblogs.com/yoyogis/archive/2012/04/19/2456724.html
	 * args:
	 * @trainfile:训练文件
	 * @testfile:测试文件
	 * @classindex:类别属性所在的列,默认是最后一列
	 */
	public void classifylinear(String trainfile, String testfile, int classindex) throws Exception {
		// 训练分类模型
		LinearRegression classifier = new LinearRegression();//线性回归
		Instances traininstance = loadarffdata(trainfile, classindex);// 获取训练数据实例
		classifier.buildClassifier(traininstance);
		System.out.println("linear regression model train over!!");
		
		//打印出训练得到的参数
		/*double[] coef = classifier.coefficients();
		for(int i = 0; i < coef.length; i++){
			System.out.println(coef[i]);
		}*/
		
		Evaluation eval = new Evaluation(traininstance);
		Instances testinstance = loadarffdata(testfile, classindex);
		eval.evaluateModel(classifier, testinstance);
		
		//分类预测
		//方法一：将预测结果以数组的形式返回
		/*double res[]=eval.evaluateModel(classifier, testinstance);
		for(int j = 0; j < res.length; j++){
			System.out.println(res[j]);
		}*/
		
		//方法二：调用分类器方法classifyInstance（Instance instance）循环输出每一个实例的预测值
		double  sum = testinstance.numInstances();//获取预测实例的总数
		for(int i=0;i<sum;i++){
			System.out.println(testinstance.instance(i).value(0)+" : "+classifier.classifyInstance(testinstance.instance(i)));
		}
		
		System.out.println("linear regression predict is over!!");
	}
	
	

	public void printInstance(Instances instances) {
		// 获取属性总数
		System.out.println("attributes num:" + instances.numAttributes());
		// 遍历行数据
		System.out.println("------all data------");
		for (int i = 0; i < instances.numInstances(); i++) {
			System.out.println(instances.instance(i));// 遍历每行数据
			// System.out.println(instances.instance(i).toString(0));//打印第一列属性
		}
	}
	

	public static void main(String[] args) throws Exception {
		//logistic 测试
		int logisticclassindex = 1;
		Regression reg = new Regression();
		String logistictrainfile = "./data/logistictrain.arff";
		String logistictestfile = "./data/logistictest.arff";
		reg.classifylogistic(logistictrainfile, logistictestfile, logisticclassindex);
		System.out.println();
		
		//linear 测试
		int linerclassindex = 1;
		String lineartrainfile = "./data/lineartrain.arff";
		String lineartestfile = "./data/lineartest.arff";
		reg.classifylinear(lineartrainfile, lineartestfile, linerclassindex);
	}

}

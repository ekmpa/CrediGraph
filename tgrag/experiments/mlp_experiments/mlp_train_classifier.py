import argparse
import pickle
import numpy as np
from tgrag.utils.path import get_root_dir
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as  Sklearn_MLPClassifier
from mlp_modules import MLPRegressor
from mlp_modules import MLP3LayersPredictor
from mlp_modules import MultiTaskMLP,train_scikitlearn_classifier,LabelPredictor,train_classifier_unbalanced
from sklearn.preprocessing import normalize 
from utils import train_valid_test_split,resize_emb,\
                  plot_loss,plot_regression_scatter,eval,\
                  plot_classesCount,plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from dataset_loader import DomainRel
import logging
from tgrag.utils.logger import setup_logging

def write_testset_emb(args,X_test,X_test_feat):
    test_set_emb_dict=dict(zip(X_test["domain"].tolist(),X_test_feat))
    test_emb_file_name=f"{args.plots_out_path}/weaksupervision_{args.month}_{args.embed_type}_{args.emb_model}_{f'GAT-{args.gnn_encoder}' if args.use_gnn_emb else ''}{'-'+args.agg_function+'_' if args.agg_month_emb else ''}test_set_emb_dict.pkl"
    with open(test_emb_file_name, 'wb') as file:
        pickle.dump(test_set_emb_dict, file)
def mlp_classifier(args) -> None:
    run_file_name=f"DomainRel_{args.month}_{args.domainRel_target}_{args.library}_{args.embed_type}_{args.emb_model}_{f'GAT-{args.gnn_encoder}' if args.use_gnn_emb else ''}{'-'+args.agg_function if args.agg_month_emb else ''}{'_topic-emb' if args.use_topic_emb else ''}"
    setup_logging(f"{args.logs_out_path}/{run_file_name}.log")          
    logging.info(f"args={args}")  
    run_file_name=f"{args.plots_out_path}/{run_file_name}"
    ############################
    X_train, y_train, X_valid, y_valid, X_test, y_test,X_train_feat, X_valid_feat, X_test_feat=DomainRel.load_run_embeddings(args)
    ############## Save Test set Embeddings dict pickle #############
    write_testset_emb(args,X_test,X_test_feat)
    ################# Train #####################
    results = []
   
    for i in range(args.runs):        
        logging.info(f"#################### Run {i} ##################")    
        run_file_name+=f"_run{str(i)}"   
        ###################### PYTorch Regressor  ######################
        if args.library=="pytorch":
            mlp_clf = LabelPredictor(len(X_train_feat[0]))
            # mlp_clf = MLPRegressor(len(X_train_feat[0]),hidden_layer_sizes=(int(len(X_train_feat[0]) * 0.75), 128, 64, 16))                
            mlp_clf, train_loss, valid_loss, test_loss, mean_loss = train_classifier_unbalanced(mlp_clf, X_train_feat, y_train,
                                                                            X_valid_feat, y_valid, X_test_feat,
                                                                            y_test, epochs=args.epochs)
            
            # mlp_clf, train_loss, valid_loss, test_loss, mean_loss = train_classifier(mlp_clf, X_train_feat, y_train,
            #                                                                 X_valid_feat, y_valid, X_test_feat,
            #                                                                 y_test, epochs=args.epochs)
        ######################## Scikit-Learn ###################
        elif args.library=="sklearn":
            mlp_clf = Sklearn_MLPClassifier(hidden_layer_sizes=(128, 32),
                                activation='relu', solver='adam',max_iter=args.max_iter, random_state=42,
                                verbose=False, learning_rate_init=args.lr,warm_start=True)
            mlp_clf.out_activation_ = 'sigmoid'
            mlp_clf, train_loss, valid_loss, test_loss, mean_loss = train_scikitlearn_classifier(mlp_clf, X_train_feat, y_train,
                                                                            X_valid_feat, y_valid, X_test_feat,
                                                                            y_test, epochs=args.epochs)
                                                                              
       
        ###################### Save Model ####################
        run_file_name=f"{args.plots_out_path}/weaksupervision_{args.month}_{args.domainRel_target}_{args.library}_{args.embed_type}_{args.emb_model}_{f'GAT-{args.gnn_encoder}' if args.use_gnn_emb else ''}{'-'+args.agg_function if args.agg_month_emb else ''}{'_topic-emb' if args.use_topic_emb else ''}_run{str(i)}"
        with open(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_credibench_MLP_Model.pkl", 'wb') as file:
            pickle.dump(mlp_clf, file)        
        ################## Plot and Eval ###############
        acc_lst,f1_lst=[],[]
        true = y_test
        pred = mlp_clf.predict(X_test_feat)
        pred=pred.round()
        accuracy = accuracy_score(true, pred)
        acc_lst.append(accuracy)
        logging.info(f"Accuracy: {accuracy:.4f}")
        cm = confusion_matrix(true, pred)
        logging.info(f"cm: {cm}")        
        f1 = f1_score(true,pred, average='macro')
        f1_lst.append(f1)
        logging.info(f"F1 score (macro): {f1:.4f}")
        plot_loss(train_loss, valid_loss, test_loss, mean_loss, run_file_name+"_loss.pdf",ylabel="CrossEntropy Loss")
        plot_classesCount(cm,run_file_name+"_class_frequancy.pdf")
        plot_regression_scatter(true, pred, run_file_name+"_testset_true_vs_pred_scatter.pdf")
        plot_confusion_matrix(cm, run_file_name+"_testset_confusion_matrix.pdf")

    logging.info(f"ACC MAEN={np.mean(acc_lst)}")
    logging.info(f"ACC STD={np.std(acc_lst)}")
    results_df = pd.DataFrame(list(zip(acc_lst,f1_lst,[args,args,args])), columns=['ACC', 'F1', 'args'])
    results_df.to_csv(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_results.csv",index=None)
    
    


if __name__ == '__main__':
    root = str(get_root_dir())
    parser = argparse.ArgumentParser(description="MLP Experiments")
    parser.add_argument("--domainRel_target", type=str, default="weak_label", choices=["weak_label"], help="the credability target")
    parser.add_argument("--domainRel_text_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--domainRel_gnn_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--domainRel_path", type=str, default=str(root + "/data/weaksupervision"),help="dqr dataset path")
    parser.add_argument("--embed_type", type=str, default="text",
                        choices=["text", "domainName", "GNN_GAT", "TFIDF", "PASTEL"], help="domains embedding technique")
    parser.add_argument("--emb_model", type=str, default="Qwen3-Embedding-0.6B",
                        choices=["Qwen3-Embedding-8B", "Qwen3-Embedding-0.6B", "embeddinggemma-300m", "TE3L","Qwen3-Embedding-8B-Q5_K_M",
                                 "IPTC_Topic_emb"],help="LLM embedding model")
    parser.add_argument("--batch_size", type=int, default=4000,help="training batch size")
    parser.add_argument("--test_valid_size", type=float, default=0.4,help="ratio of test and vaild sets")
    parser.add_argument("--emb_dim", type=int, default=1024,help="embedding size")
    parser.add_argument("--original_emb_dim", type=int, default=1024,help="The original embedding model dim size")
    parser.add_argument("--max_iter", type=int, default=200,help="MLP regressor max iteration count")
    parser.add_argument("--lr", type=float, default=1e-1,help="learning rate")
    parser.add_argument("--epochs", type=int,default=200, help="# training epochs") 
    parser.add_argument("--plots_out_path", type=str, default=str(root + "/plots"),help="plots and results store path")
    parser.add_argument("--logs_out_path", type=str, default=str(root + "/logs"),help="logging path")
    parser.add_argument("--runs", type=int, default=1,help="# training runs")
    parser.add_argument("--use_gnn_emb", action='store_false',help="append GNN embedding")
    parser.add_argument("--gnn_encoder",   default="text", choices=["RNI","text"],help="append GNN node embedding intialization")
    parser.add_argument("--agg_month_emb",  action='store_false', help="aggregate montly GNN embeddings")
    parser.add_argument("--agg_text_emb", action='store_false', help="aggregate montly text embeddings")
    parser.add_argument("--agg_function", type=str, default="avg",choices=["avg","cat"], help="aggregate function")
    parser.add_argument("--use_topic_emb", action='store_true',help="use topic modeling features")
    parser.add_argument("--filter_by_GNN_nodes", action='store_false',help="filter by domains has GNN embeddings")
    parser.add_argument("--num_classes", type=int, default=2,help="# classifcation classes for Multihead model")
    parser.add_argument("--generate_weaksupervision_scores", type=bool, default=False, help="gnerate weak supervision datasets scores")
    parser.add_argument("--month", type=str, default="dec", choices=["oct", "nov", "dec"],help="CrediBench month snapshot")
    parser.add_argument("--library", type=str, default="pytorch", choices=["pytorch", "sklearn"],help="ML library to use")
    args = parser.parse_args()
    mlp_classifier(args)
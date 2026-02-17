import argparse
import pickle
import logging
from tgrag.utils.path import get_root_dir
from mlp_modules import MultiTaskMLP, train_multihead_unbalanced
from utils import plot_loss,plot_regression_scatter,eval,plot_classesCount,plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from dataset_loader import DQR,DomainRel
import sys
from tgrag.utils.logger import setup_logging
def mlp_train_multihead(args) -> None:
    learnable_loss_weight=True
    run_file_name=f"multihead_{args.month}_{args.dqr_target}_{args.library}_{args.embed_type}_{args.emb_model}_{f'GAT-{args.gnn_encoder}' if args.use_gnn_emb else ''}{'-'+args.agg_function if args.agg_month_emb else ''}{'_LLW' if learnable_loss_weight else 'MLW'}"
    setup_logging(f"{args.logs_out_path}/{run_file_name}.log",logging.INFO,logging.INFO)    
    logging.info(f"args={args}")
    # setup_logging(f"{run_file_name}.log",logging.INFO,logging.INFO)    
    run_file_name=f"{args.plots_out_path}/{run_file_name}"
    
    ############## Load  training data and split ###############
    reg_X_train, reg_y_train, reg_X_valid, reg_y_valid, reg_X_test, reg_y_test,reg_X_train_feat, reg_X_valid_feat, reg_X_test_feat=DQR.load_run_embeddings(args)
    clf_X_train, clf_y_train, clf_X_valid, clf_y_valid, clf_X_test, clf_y_test,clf_X_train_feat, clf_X_valid_feat, clf_X_test_feat=DomainRel.load_run_embeddings(args)   
    ################# Train #####################   
    for i in range(args.runs):
        logging.info(f"#################### Run {i} ##################")    
        run_file_name+=f"_run{str(i)}"        
        mlp_MT = MultiTaskMLP(input_dim=len(reg_X_train_feat[0]), num_classes=len(set(clf_y_train)),
                                   hidden_dims=[len(clf_X_train_feat[0]),len(clf_X_train_feat[0])//4])            
        mlp_MT, train_loss,train_loss_clf, valid_loss,valid_loss_clf, test_loss,test_loss_clf, mean_loss = train_multihead_unbalanced(mlp_MT, reg_X_train_feat, reg_y_train,reg_X_valid_feat, reg_y_valid, reg_X_test_feat,reg_y_test,
                                                                                        clf_X_train_feat, clf_y_train,clf_X_valid_feat, clf_y_valid, clf_X_test_feat,clf_y_test,
                                                                                  lr=args.lr,epochs=args.epochs,batch_size=args.batch_size,use_URL_Loss= learnable_loss_weight)
        ###################### Save Model ####################                
        with open(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_credibench_MLP_Model.pkl", 'wb') as file:
            pickle.dump(mlp_MT, file)        
        ################## Plot and Eval ###############
        reg_true = reg_y_test
        mlp_MT.eval()
        reg_pred = mlp_MT.predict(reg_X_test_feat,task="reg").detach()
        clf_pred = mlp_MT.predict(clf_X_test_feat,task="cls").detach()
        MSE, MAE, R2, Mean_MAE, min_error_dict, max_error_dict = eval(reg_pred, reg_true)
        logging.info(f"Regression: MSE={MSE}\tMAE={MAE}\tR2={R2}\tMean_MAE={Mean_MAE}" )
        clf_pred=clf_pred.round()
        clf_true = clf_y_test
        accuracy = accuracy_score(clf_true, clf_pred)
        logging.info(f"Binary DS: Accuracy= {accuracy:.4f}")
        cm = confusion_matrix(clf_true, clf_pred)
        logging.info(f"Binary DS:CM= {cm}")        
        f1 = f1_score(clf_true,clf_pred, average='macro')
        logging.info(f"Binary DS F1 score (macro): {f1:.4f}")
        plot_loss(train_loss, valid_loss, test_loss, mean_loss, run_file_name+"_multihead_reg_loss.pdf",ylabel="MSE")
        plot_loss(train_loss_clf, valid_loss_clf, test_loss_clf, None, run_file_name+"_multihead_clf_loss.pdf",ylabel="CrossEntropy Loss")
        plot_classesCount(cm,run_file_name+"_class_frequancy.pdf")
        plot_regression_scatter(clf_true, clf_pred, run_file_name+"_testset_multihead_true_vs_pred_scatter.pdf")
        plot_confusion_matrix(cm, run_file_name+"_testset_multihead_confusion_matrix.pdf")     

if __name__ == '__main__':
    root = str(get_root_dir())
    parser = argparse.ArgumentParser(description="MLP Experiments")
    parser.add_argument("--domainRel_target", type=str, default="weak_label", choices=["weak_label"], help="the credability target")
    parser.add_argument("--dqr_target", type=str, default="pc1", choices=["pc1","mbfc"], help="the credability target")
    parser.add_argument("--domainRel_text_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--domainRel_gnn_emb_path", type=str, default=str(root + "/data/weaksupervision") ,help="emb files path")
    parser.add_argument("--domainRel_path", type=str, default=str(root + "/data/weaksupervision"),help="dqr dataset path")
    parser.add_argument("--dqr_text_emb_path", type=str, default=str(root + "/data/dqr") ,help="emb files path")
    parser.add_argument("--dqr_gnn_emb_path", type=str, default=str(root + "/data/dqr") ,help="emb files path")
    parser.add_argument("--dqr_path", type=str, default=str(root + "/data/dqr"),help="dqr dataset path")
    parser.add_argument("--embed_type", type=str, default="text",
                        choices=["text", "domainName", "GNN_GAT", "TFIDF", "PASTEL"], help="domains embedding technique")
    parser.add_argument("--emb_model", type=str, default="embeddingTE3L",
                        choices=["embeddingQwen3-8B", "embeddingQwen3-0.6B", "embeddinggemma-300m", "embeddingTE3L",
                                 "IPTC_Topic_emb"],help="LLM embedding model")
    parser.add_argument("--batch_size", type=int, default=4000,help="training batch size")
    parser.add_argument("--test_valid_size", type=float, default=0.4,help="ratio of test and vaild sets")
    parser.add_argument("--emb_dim", type=int, default=256,help="embedding size")
    parser.add_argument("--original_emb_dim", type=int, default=1024,help="The original embedding model dim size")
    parser.add_argument("--max_iter", type=int, default=200,help="MLP regressor max iteration count")
    parser.add_argument("--lr", type=float, default=1e-4,help="learning rate")
    parser.add_argument("--epochs", type=int,default=200, help="# training epochs")  
    parser.add_argument("--plots_out_path", type=str, default=str(root + "/plots"),help="plots and results store path")
    parser.add_argument("--logs_out_path", type=str, default=str(root + "/logs"),help="logging path")
    parser.add_argument("--runs", type=int, default=1,help="# training runs")
    parser.add_argument("--gnn_encoder",   default="text", choices=["RNI","text"],help="append GNN node embedding intialization")
    parser.add_argument("--use_gnn_emb", action='store_false',help="append GNN embedding")
    parser.add_argument("--agg_text_emb", action='store_false', help="aggregate montly text embeddings")
    parser.add_argument("--agg_month_emb",action='store_false', help="aggregate montly GNN embeddings")
    parser.add_argument("--agg_function", type=str, default="avg",choices=["avg","cat"], help="aggregate function")
    parser.add_argument("--filter_by_GNN_nodes", action='store_false',help="filter by domains has GNN embeddings")
    parser.add_argument("--num_classes", type=int, default=2,help="# classifcation classes for Multihead model")
    parser.add_argument("--month", type=str, default="dec", choices=["oct", "nov", "dec"],help="CrediBench month snapshot")
    parser.add_argument("--library", type=str, default="pytorch", choices=["pytorch", "sklearn"],help="ML library to use")
    args = parser.parse_args()
    mlp_train_multihead(args)
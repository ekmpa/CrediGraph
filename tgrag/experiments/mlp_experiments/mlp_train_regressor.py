import argparse
import pickle
from tgrag.utils.path import get_root_dir
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor as  Sklearn_MLPRegressor
from mlp_modules import MLPRegressor
from mlp_modules import MLP3LayersPredictor
from mlp_modules import MultiTaskMLP,train_mlp,train_scikitlearn_regressor
from utils import plot_histogram,plot_loss,plot_regression_scatter,eval
from dataset_loader import DQR
from tgrag.utils.logger import setup_logging
import logging
def mlp_train(args) -> None:
    run_file_name=f"dqr_{args.month}_{args.dqr_target}_{args.library}_{args.embed_type}_{args.emb_model}_{f'GAT-{args.gnn_encoder}' if args.use_gnn_emb else ''}{'-'+args.agg_function if args.agg_month_emb else ''}{'_topic-emb' if args.use_topic_emb else ''}"
    setup_logging(f"{args.logs_out_path}/{run_file_name}.log")      
    logging.info("args=", args)      
    run_file_name=f"{args.plots_out_path}/{run_file_name}"
    ############################################
    X_train, y_train, X_valid, y_valid, X_test, y_test,X_train_feat, X_valid_feat, X_test_feat=DQR.load_run_embeddings(args)
    results = []        
    for i in range(args.runs):
        logging.info(f"#################### Run {i} ##################")    
        run_file_name+=f"_run{str(i)}" 
        
        ###################### PYTorch Regressor  ######################
        if args.library=="pytorch":
            mlp_reg = MLP3LayersPredictor(len(X_train_feat[0]))
            # mlp_reg = MLPRegressor(len(X_train_feat[0]),hidden_layer_sizes=(int(len(X_train_feat[0]) * 0.75), 128, 64, 16))
            mlp_reg, train_loss, valid_loss, test_loss, mean_loss = train_mlp(mlp_reg, X_train_feat, y_train,
                                                                            X_valid_feat, y_valid, X_test_feat,
                                                                            y_test, epochs=args.epochs)
        ######################## Scikit-Learn ###################
        elif args.library=="sklearn":
            mlp_reg = Sklearn_MLPRegressor(hidden_layer_sizes=(128, 32),
                                activation='relu', solver='adam',max_iter=args.max_iter, random_state=42,
                                verbose=False, learning_rate_init=args.lr)
            mlp_reg.out_activation_ = 'sigmoid'
            mlp_reg, train_loss, valid_loss, test_loss, mean_loss = train_scikitlearn_regressor(mlp_reg, X_train_feat, y_train,
                                                                            X_valid_feat, y_valid, X_test_feat,
                                                                            y_test, epochs=args.epochs)
        ###################### Save Model ####################        
        with open(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_credibench_MLP_Model.pkl", 'wb') as file:
            pickle.dump(mlp_reg, file)
        ################## Weak Supervision ###########
        if args.generate_weaksupervision_scores == True:
            if args.agg_month_emb:
                embd_dict_phishtank, embd_dict_URLhaus, embd_dict_PhishDataset_legit = DQR.load_weaksupervision_emb_dict(
                    args.embed_type, args.text_emb_path, args.emb_model, month="dec", target=args.dqr_target,
                    gnn_emb=args.use_gnn_emb, agg="avg")
            else:
                embd_dict_phishtank, embd_dict_URLhaus, embd_dict_PhishDataset_legit = DQR.load_weaksupervision_emb_dict(
                    args.embed_type, args.text_emb_path, args.emb_model, month="dec", target=args.dqr_target,
                    gnn_emb=args.use_gnn_emb, agg=None)

            phishtank_features = [v for k, v in embd_dict_phishtank.items()]
            phishtank_pred = mlp_reg.predict(phishtank_features)
            pd.DataFrame(zip(embd_dict_phishtank.keys(), phishtank_pred),
                         columns=["domain", f"pred_{args.dqr_target}"]).to_csv(
                        f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_Phishtank_pred.csv",index=None)
            # logging.info("phishtank_pred=",phishtank_pred)

            URLhaus_features = [v for k, v in embd_dict_URLhaus.items()]
            URLhaus_pred = mlp_reg.predict(URLhaus_features)
            pd.DataFrame(zip(embd_dict_URLhaus.keys(), URLhaus_pred), columns=["domain", f"pred_{args.dqr_target}"]).to_csv(
                f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_URLhaus_pred.csv",index=None)

            PhishDataset_legit_features = [v for k, v in embd_dict_PhishDataset_legit.items()]
            PhishDataset_legit_pred = mlp_reg.predict(PhishDataset_legit_features)
            pd.DataFrame(zip(embd_dict_PhishDataset_legit.keys(), PhishDataset_legit_pred),
                         columns=["domain", f"pred_{args.dqr_target}"]).to_csv(
                        f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_legit_pred.csv",index=None)
        ################## Plot and Eval ###############
        true = y_test
        pred = mlp_reg.predict(X_test_feat) 
        
        plot_loss(train_loss, valid_loss, test_loss, mean_loss, run_file_name+"_loss.pdf")
        plot_histogram(true, pred,  run_file_name+"_testset_true_vs_pred_frequency.pdf")
        plot_regression_scatter(true, pred, run_file_name+"_testset_true_vs_pred_scatter.pdf")
        MSE, MAE, R2, Mean_MAE, min_error_dict, max_error_dict = eval(pred, true)
        min_error_dict["domain"] = X_test.iloc[min_error_dict["test_idx"]]["domain"]
        max_error_dict["domain"] = X_test.iloc[max_error_dict["test_idx"]]["domain"]
        results.append([MSE, MAE, R2, Mean_MAE, str(min_error_dict), str(max_error_dict), str(args)])
        logging.info(
            f"Run{i}:MSE={MSE}\tR2={R2}\tMAE={MAE}\tMean_MAE={Mean_MAE}\tmin_error_dict={min_error_dict}\tmax_error_dict={max_error_dict}")
        ############ save test results ############
        X_test.to_csv(
            f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_testset.csv",index=None)
        pd.DataFrame(zip(true, pred), columns=["true", "pred"]).to_csv(
            f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_pred.csv",index=None)

    results_df = pd.DataFrame(results, columns=['MSE', 'MAE', 'R2', 'Mean_MAE', 'Min_AE', 'Max_AE', 'args'])
    results_df.to_csv(f"{run_file_name}{'_agg' if args.agg_month_emb else ''}_dqr_results.csv",index=None)
    
    for col in ['MSE', 'MAE', 'R2', 'Mean_MAE']:
        logging.info(f"{col}: Mean={results_df[col].mean()}\tstd={results_df[col].std()}")


if __name__ == '__main__':
    root = str(get_root_dir())
    parser = argparse.ArgumentParser(description="MLP Experiments")
    parser.add_argument("--dqr_target", type=str, default="pc1", choices=["pc1", "mbfc", "mbfc_bias"], help="the credability target")
    parser.add_argument("--dqr_text_emb_path", type=str, default=str(root + "/data/dqr") ,help="emb files path")
    parser.add_argument("--dqr_gnn_emb_path", type=str, default=str(root + "/data/dqr") ,help="emb files path")
    parser.add_argument("--dqr_path", type=str, default=str(root + "/data/dqr"),help="dqr dataset path")
    parser.add_argument("--embed_type", type=str, default="text",
                        choices=["text", "domainName", "GNN_GAT", "TFIDF", "PASTEL"], help="domains embedding technique")
    parser.add_argument("--emb_model", type=str, default="embeddingTE3L",
                        choices=["embeddingQwen3-8B", "embeddingQwen3-0.6B", "embeddinggemma-300m", "embeddingTE3L",
                                 "IPTC_Topic_emb"],help="LLM embedding model")
    parser.add_argument("--batch_size", type=int, default=5000,help="training batch size")
    parser.add_argument("--test_valid_size", type=float, default=0.4,help="ratio of test and vaild sets")
    parser.add_argument("--emb_dim", type=int, default=3072,help="embedding size")
    parser.add_argument("--original_emb_dim", type=int, default=3072,help="The original embedding model dim size")
    parser.add_argument("--max_iter", type=int, default=200,help="MLP regressor max iteration count")
    parser.add_argument("--lr", type=float, default=5e-3,help="learning rate")
    parser.add_argument("--epochs", type=int,default=100, help="# training epochs")  
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
    parser.add_argument("--filter_by_PASTEL_domains", action='store_true',help="filter by domains has PASTEL embeddings")
    parser.add_argument("--generate_weaksupervision_scores", type=bool, default=False, help="gnerate weak supervision datasets scores")
    parser.add_argument("--month", type=str, default="dec", choices=["oct", "nov", "dec"],help="CrediBench month snapshot")
    parser.add_argument("--library", type=str, default="pytorch", choices=["pytorch", "sklearn"],help="ML library to use")
    args = parser.parse_args()
    mlp_train(args)
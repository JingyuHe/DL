########## Data Preparation ##########
# Load in data
load('../data/da_197201_202112_ins_finaluse.rda')
load('../data/da_197201_202112_oos_finaluse.rda')

library(stringr)
library(MASS)

# Split dataset
start = '1972-01-01'
split = '2011-12-31'
end = '2021-12-31'

data = rbind(data1, data2); rm("data1","data2")
data = data[order(data$date), ]
data = data[,append(c(1:92), 103)]

char_list = c("rank_me","rank_bm","rank_agr","rank_op")
ff5 = c('MKTRF','SMB','HML','RMW','CMA')
ipca = c('IPCA_1','IPCA_2','IPCA_3','IPCA_4','IPCA_5')
rppca = c('RPPCA_1','RPPCA_2','RPPCA_3','RPPCA_4','RPPCA_5')
factor_list = c(ff5, ipca, rppca)
model_list = c('CAPM','FF5','IPCA','RPPCA')
monthlist = sort(unique(data$date))

df_factor = read.csv('../data/raw_data_generate/addition_factors_20220906.csv')

df_ipca_ins = read.csv('../data/ipca5_ins_1972-01-31_19720202.csv'); colnames(df_ipca_ins) = c("date", ipca)
df_ipca_oos = read.csv('../data/ipca5_oos_1972-01-31_19720202.csv'); colnames(df_ipca_oos) = c("date", ipca)
df_ipca = rbind(df_ipca_ins, df_ipca_oos)
df_ipca$date = monthlist
data <- merge(data, df_ipca, by="date", all.x = T)

df_rppca_ins = read.csv('../data/rppca_ins_uni_40_10.csv', header = FALSE); colnames(df_rppca_ins) = rppca
df_rppca_oos = read.csv('../data/rppca_oos_uni_40_10.csv', header = FALSE); colnames(df_rppca_oos) = rppca
df_rppca = rbind(df_rppca_ins, df_rppca_oos)
df_rppca$date = monthlist
data <- merge(data, df_rppca, by="date", all.x = T)

data_capm = as.matrix(df_factor[,c("MKTRF")])
data_ff5 = df_factor[,ff5]
data_ipca = df_ipca[,ipca]
data_rppca = df_rppca[,rppca]

layer_list = c(0:3)
g_dim_list = c(1,5)
l1_v_list = c(-5:-3)
min_n_factor_list = c(3:5)

finaluse_list = c('FF5','IPCA','RPPCA',
                  'factors1_0_5_-3_cv2','factors5_0_5_-5_cv2','factors1_1_5_-5_cv2','factors5_1_5_-4_cv2',
                  'factors1_2_5_-3_cv2','factors5_2_5_-4_cv2','factors1_3_5_-3_cv2','factors5_3_5_-5_cv2')

data1 <- data[(data[,c('date')]>=start) & (data[,c('date')]<=split), ]
data2 <- data[(data[,c('date')]>split) & (data[,c('date')]<=end), ]

judge.star <- function(value){
  if (value > 0.99){
    output = "***"
  } else if (value > 0.95){
    output = "**"
  } else if (value > 0.90){
    output = "*"
  } else{
    output = ""
  }
  return(output)
}


########## Tangency Portfolio Sharpe Ratio ##########
tpsr.calc <- function(df, df_bench){
  df_ins = as.matrix(df[1:480,])
  df_oos = as.matrix(df[481:600,])
  df_bench_ins = as.matrix(df_bench[1:480,])
  df_bench_oos = as.matrix(df_bench[481:600,])
  
  E <- apply(df_ins, 2, mean)
  V <- cov(df_ins)
  inV <- solve(V)
  oneH <- matrix(rep(c(1), dim(df_ins)[2]), c(1,dim(df_ins)[2]))
  wTP <- (inV%*%E)/(oneH%*%inV%*%E)[1,1]
  tp_rtn_ins <- as.matrix(df_ins)%*%as.matrix(wTP)
  tp_rtn_oos <- as.matrix(df_oos)%*%as.matrix(wTP)
  if (mean(tp_rtn_ins) <= 0){
    tp_rtn_ins = tp_rtn_ins*(-1)
    tp_rtn_oos = tp_rtn_oos*(-1)
  } else{
    tp_rtn_ins = tp_rtn_ins
    tp_rtn_oos = tp_rtn_oos
  }
  
  tp_sr_ins <- mean(tp_rtn_ins)/sd(tp_rtn_ins)*sqrt(12)
  tp_sr_oos <- mean(tp_rtn_oos)/sd(tp_rtn_oos)*sqrt(12)
  
  # Start to do Squared Sharpe Ratio Test in Barillas et al. (2020)
  F_t_ins = cbind(df_ins, df_bench_ins)
  E_ins = apply(F_t_ins, 2, mean)
  V_ins = cov(F_t_ins)
  inV_ins = ginv(V_ins)
  oneH_ins <- matrix(rep(c(1), dim(F_t_ins)[2]), c(1,dim(F_t_ins)[2]))
  wTP_ins <- (inV_ins%*%E_ins)/(oneH_ins%*%inV_ins%*%E_ins)[1,1]
  f_t_ins = as.matrix(F_t_ins)%*%as.matrix(wTP_ins)
  g_t_ins = as.matrix(df_bench_ins)
  ins_reg = summary(lm(f_t_ins ~ g_t_ins))
  alpha_ins = as.matrix(ins_reg$coefficients[1,1])
  sigma_epsilon_ins = as.matrix(cov(as.matrix(ins_reg$residuals)))
  g_mean_ins = as.matrix(apply(g_t_ins, 2, mean))
  sigma_g_ins = as.matrix(cov(g_t_ins))
  srg2 = t(g_mean_ins)%*%solve(sigma_g_ins)%*%g_mean_ins
  srf2_srg2 = t(alpha_ins)%*%solve(sigma_epsilon_ins)%*%alpha_ins
  T_ins = dim(F_t_ins)[1]; P_ins = dim(df_ins)[2]; D_ins = dim(df_bench)[2]
  f_value_ins = (T_ins/P_ins)*((T_ins-P_ins-D_ins)/(T_ins-D_ins-1))*srf2_srg2/(1+srg2)
  star_value_ins = pf(f_value_ins, P_ins, T_ins-P_ins-D_ins)
  judge_ins = judge.star(star_value_ins)
  
  # F_t_oos = cbind(df_oos, df_bench_oos)
  # E_oos = apply(F_t_oos, 2, mean)
  # V_oos = cov(F_t_oos)
  # inV_oos = ginv(V_oos)
  # oneH_oos <- matrix(rep(c(1), dim(F_t_oos)[2]), c(1,dim(F_t_oos)[2]))
  # wTP_oos <- (inV_oos%*%E_oos)/(oneH_oos%*%inV_oos%*%E_oos)[1,1]
  # f_t_oos = as.matrix(F_t_oos)%*%as.matrix(wTP_oos)
  # g_t_oos = as.matrix(df_bench_oos)
  # oos_reg = summary(lm(f_t_oos ~ g_t_oos))
  # alpha_oos = as.matrix(oos_reg$coefficients[1,1])
  # sigma_epsilon_oos = as.matrix(cov(as.matrix(oos_reg$residuals)))
  # g_mean_oos = as.matrix(mean(g_t_oos))
  # sigma_g_oos = as.matrix(cov(g_t_oos))
  # srg2 = t(g_mean_oos)%*%solve(sigma_g_oos)%*%g_mean_oos
  # srf2_srg2 = t(alpha_oos)%*%solve(sigma_epsilon_oos)%*%alpha_oos
  # T_oos = dim(F_t_oos)[1]; P_oos = dim(df_oos)[2]; D_oos = dim(df_bench)[2]
  # f_value_oos = (T_oos/P_oos)*((T_oos-P_oos-D_oos)/(T_oos-D_oos-1))*srf2_srg2/(1+srg2)
  # star_value_oos = pf(f_value_oos, P_oos, T_oos-P_oos-D_oos)
  # judge_oos = judge.star(star_value_oos)
  
  # return(c(paste0(sprintf("%0.2f", tp_sr_ins), judge_ins), paste0(sprintf("%0.2f", tp_sr_oos), judge_oos)))
  return(c(paste0(sprintf("%0.2f", tp_sr_ins), judge_ins), sprintf("%0.2f", tp_sr_oos)))
}

sr_list = c(); sr_value_ins = c(); sr_value_oos = c()
n_layer_list = c(); g_list = c(); regularization_list = c(); n_factor_list = c(); min_n_factor = c()

capm_sr = tpsr.calc(data_capm, data_capm)
sr_value_ins = append(sr_value_ins, capm_sr[1]); sr_value_oos = append(sr_value_oos, capm_sr[2])
sr_list = append(sr_list, "CAPM")
n_layer_list = append(n_layer_list, " "); g_list = append(g_list, " "); regularization_list = append(regularization_list, " "); n_factor_list = append(n_factor_list, " "); min_n_factor = append(min_n_factor, " ")

ff5_sr = tpsr.calc(data_ff5, data_capm)
sr_value_ins = append(sr_value_ins, ff5_sr[1]); sr_value_oos = append(sr_value_oos, ff5_sr[2])
sr_list = append(sr_list, "FF5")
n_layer_list = append(n_layer_list, " "); g_list = append(g_list, " "); regularization_list = append(regularization_list, " "); n_factor_list = append(n_factor_list, " "); min_n_factor = append(min_n_factor, " ")

ipca_sr = tpsr.calc(data_ipca, data_capm)
sr_value_ins = append(sr_value_ins, ipca_sr[1]); sr_value_oos = append(sr_value_oos, ipca_sr[2])
sr_list = append(sr_list, "IPCA")
n_layer_list = append(n_layer_list, " "); g_list = append(g_list, " "); regularization_list = append(regularization_list, " "); n_factor_list = append(n_factor_list, " "); min_n_factor = append(min_n_factor, " ")

rppca_sr = tpsr.calc(data_rppca, data_capm)
sr_value_ins = append(sr_value_ins, rppca_sr[1]); sr_value_oos = append(sr_value_oos, rppca_sr[2])
sr_list = append(sr_list, "RPPCA")
n_layer_list = append(n_layer_list, " "); g_list = append(g_list, " "); regularization_list = append(regularization_list, " "); n_factor_list = append(n_factor_list, " "); min_n_factor = append(min_n_factor, " ")

for (l in 1:length(layer_list)){
  for (g in 1:length(g_dim_list)){
    for (v in 1:length(l1_v_list)){
      for (m in 1:length(min_n_factor_list)){
        filename = paste0('factors', g_dim_list[g], '_', layer_list[l], '_5_', l1_v_list[v], '_cv2'); sr_list = append(sr_list, filename)
        data_dp = read.csv(paste0('results/', filename, '.csv'))
        n_layer_list = append(n_layer_list, layer_list[l]); g_list = append(g_list, g_dim_list[g]); regularization_list = append(regularization_list, l1_v_list[v]); n_factor_list = append(n_factor_list, dim(data_dp)[2])
        min_factors = min_n_factor_list[m]; min_n_factor = append(min_n_factor, min_factors)
        data_dp = data_dp[,c(1:min_factors)]
        
        if (g_dim_list[g] == 1){
          data_dp = cbind(data_dp, data_capm)
          dp_sr = tpsr.calc(data_dp, data_capm)
        } else if (g_dim_list[g] == 5){
          data_dp = cbind(data_dp, data_ff5)
          dp_sr = tpsr.calc(data_dp, data_ff5)
        }
        
        sr_value_ins = append(sr_value_ins, dp_sr[1]); sr_value_oos = append(sr_value_oos, dp_sr[2])
      }
    }
  }
}

total_sr = cbind(sr_list, g_list, n_layer_list, n_factor_list, regularization_list, min_n_factor, sr_value_ins, sr_value_oos)
colnames(total_sr) = c('model_list', 'g', 'n_layer', 'n_factor', 'regularization', 'min_n_factor', 'ins_sr', 'oos_sr')
total_sr = data.frame(total_sr)
# total_sr = total_sr[(total_sr$model_list %in% append('CAPM', finaluse_list)),]
write.csv(data.frame(total_sr), 'tangency_port_sr.csv')









########## Calculate R2 ##########

##### Some Preparation #####
# Define functions to calculate Total R2 and Predictive R2
R2.calc = function(y, ypred, ypred_capm){
  return(100*(1 - sum((y - ypred)^2) / sum((y - ypred_capm)^2)))
}

# Define functions to calculate Cross-Sectional R2
CSR2.calc = function(stocks_index, y, haty, haty_capm){
  df <- data.frame(cbind(stocks_index, y, haty, haty_capm))
  colnames(df) = c('stocks_index','y','haty','haty_capm')
  df$e=as.numeric(df$y)-as.numeric(df$haty)
  df$ecapm = as.numeric(df$y) - as.numeric(df$haty_capm)
  numer <- aggregate(x=df$e, by=list(stocks_index), FUN=mean)$x
  denom <- aggregate(x=df$ecapm, by=list(stocks_index), FUN=mean)$x
  return((1-mean(numer**2)/mean(denom**2))*100)
}

# Calculate INS factors mean for predictive R2
insaddfactor <- data1[,c("date",factor_list)]
insaddfactor <- insaddfactor[!duplicated(insaddfactor, fromLast = TRUE),]
insaddfactor <- data.frame(insaddfactor, row.names = 1)
avg_ff_train <- colMeans(insaddfactor)

##### Individual Stocks R2 #####
instot = c(); inscs = c(); inspred = c()
oostot = c(); ooscs = c(); oospred = c()
r2_list = c()

# First calculate FF5, IPCA, RPPCA R2
for (m in 1:length(model_list)){
  r2_list = append(r2_list, model_list[m])
  y_total = c(); y_total_test = c()
  haty_total = c(); haty_total_test = c()
  haty_capm_total = c(); haty_capm_total_test = c()
  haty_total_pred = c(); haty_total_pred_test = c()
  haty_capm_total_pred = c(); haty_capm_total_pred_test = c()
  stocks_newlist = c(); stocks_newlist_test = c()

  print(paste0('Start to Calculate ', model_list[m], ' model R2'))
  model = model_list[m]

  Z_train = data1[, char_list]
  Z_test = data2[, char_list]
  R_train = data1[,c("xret")]
  R_test = data2[,c("xret")]
  stocks_newlist = append(stocks_newlist, data1$permno)
  stocks_newlist_test = append(stocks_newlist_test, data2$permno)

  if (model == 'FF5'){
    temp_model = ff5
  } else if (model == 'IPCA'){
    temp_model = ipca
  } else if (model == 'RPPCA'){
    temp_model = rppca
  } else if (model == 'CAPM'){
    temp_model = c("MKTRF")
  }

  f_t_train = as.matrix(data1[, temp_model])
  f_t_test = as.matrix(data2[, temp_model])
  avg_ff_train_new = avg_ff_train[temp_model]

  temp1_train = model.matrix(~as.matrix(Z_train):as.matrix(f_t_train) - 1)
  F_train = f_t_train
  F_train_inter = temp1_train
  rm("f_t_train", "temp1_train")

  f_t_train_pred = matrix(rep(avg_ff_train_new, dim(Z_train)[1]), nrow = dim(Z_train)[1], ncol = length(avg_ff_train_new), byrow = TRUE)
  colnames(f_t_train_pred) = temp_model
  temp1_train_pred = model.matrix(~as.matrix(Z_train):as.matrix(f_t_train_pred) - 1)
  F_train_pred = f_t_train_pred
  F_train_inter_pred = temp1_train_pred
  rm("f_t_train_pred", "temp1_train_pred")

  temp1_test = model.matrix(~as.matrix(Z_test):as.matrix(f_t_test) - 1)
  F_test = f_t_test
  F_test_inter = temp1_test
  rm("f_t_test", "temp1_test")

  f_t_test_pred = matrix(rep(avg_ff_train_new, dim(Z_test)[1]), nrow = dim(Z_test)[1], ncol = length(avg_ff_train_new), byrow = TRUE)
  colnames(f_t_test_pred) = temp_model
  temp1_test_pred = model.matrix(~as.matrix(Z_test):as.matrix(f_t_test_pred) - 1)
  F_test_pred = f_t_test_pred
  F_test_inter_pred = temp1_test_pred
  rm("f_t_test_pred", "temp1_test_pred")

  # R_{i,t} = a_0 + (b_0 + b^T*z_{i,t-1})*f_t + e_{i,t}
  # In-Sample Total R2 and Cross-Sectional R2
  y <- R_train; rm("R_train")
  x <- as.matrix(cbind(F_train, F_train_inter))

  ins_reg <- summary(lm(y ~ x))#; print(ins_reg)
  b_model <- ins_reg$coefficients[,1]
  haty <- x%*%as.matrix(b_model[2:length(b_model)])
  y_total <- append(y_total, y); rm("y")
  haty_total <- append(haty_total, haty); rm("haty")
  haty_capm_total <- append(haty_capm_total, data1[,c("MKTRF")])
  rm("x")

  # In-Sample Predictive R2
  x <- as.matrix(cbind(F_train_pred, F_train_inter_pred))
  haty <- x%*%as.matrix(b_model[2:length(b_model)])
  haty_total_pred <- append(haty_total_pred, haty); rm("haty")
  haty_capm_total_pred <- append(haty_capm_total_pred, rep(avg_ff_train[c("MKTRF")], dim(x)[1]))
  rm("x")

  # Out-of-Sample Total R2 and Cross-Sectional R2
  y <- R_test; rm("R_test")
  x <- as.matrix(cbind(F_test, F_test_inter))
  haty <- x%*%as.matrix(b_model[2:length(b_model)])
  y_total_test <- append(y_total_test, y); rm("y")
  haty_total_test <- append(haty_total_test, haty); rm("haty")
  haty_capm_total_test <- append(haty_capm_total_test, data2[,c("MKTRF")])
  rm("x")

  # Out-of-Sample Predictive R2
  x <- as.matrix(cbind(F_test_pred, F_test_inter_pred))
  haty <- x%*%as.matrix(b_model[2:length(b_model)])
  haty_total_pred_test <- append(haty_total_pred_test, haty); rm("haty")
  haty_capm_total_pred_test <- append(haty_capm_total_pred_test, rep(avg_ff_train[c("MKTRF")], dim(x)[1]))
  rm("x")


  # Start to output R2 results and save
  ins_TotR2 = R2.calc(y_total, haty_total, haty_capm_total)
  print(paste0(model_list[m], " In-Sample Total R2 %: "))
  print(ins_TotR2)
  instot = append(instot, ins_TotR2)

  ins_CSR2 = CSR2.calc(stocks_newlist, y_total, haty_total, haty_capm_total)
  print(paste0(model_list[m], " In-Sample Cross-Sectional R2 %: "))
  print(ins_CSR2)
  inscs = append(inscs, ins_CSR2)

  ins_PredR2 = R2.calc(y_total, haty_total_pred, haty_capm_total_pred)
  print(paste0(model_list[m], " In-Sample Predictive R2 %: "))
  print(ins_PredR2)
  inspred = append(inspred, ins_PredR2)

  oos_TotR2 = R2.calc(y_total_test, haty_total_test, haty_capm_total_test)
  print(paste0(model_list[m], " Out-of-Sample Total R2 %: "))
  print(oos_TotR2)
  oostot = append(oostot, oos_TotR2)

  oos_CSR2 = CSR2.calc(stocks_newlist_test, y_total_test, haty_total_test, haty_capm_total_test)
  print(paste0(model_list[m], " Out-of-Sample Cross-Sectional R2 %: "))
  print(oos_CSR2)
  ooscs = append(ooscs, oos_CSR2)

  oos_PredR2 = R2.calc(y_total_test, haty_total_pred_test, haty_capm_total_pred_test)
  print(paste0(model_list[m], " Out-of-Sample Predictive R2 %: "))
  print(oos_PredR2)
  oospred = append(oospred, oos_PredR2)

}

# Start to calculate Deep Factors R2
# First get a date list to merge with dp factors
monthlist_ins = sort(unique(data1$date))
monthlist_oos = sort(unique(data2$date))
monthlist = append(monthlist_ins, monthlist_oos)

for (l in 1:length(layer_list)){
  for (g in 1:length(g_dim_list)){
    for (v in 1:length(l1_v_list)){
      for (m in 1:length(min_n_factor_list)){
        filename = paste0('factors', g_dim_list[g], '_', layer_list[l], '_5_', l1_v_list[v], '_cv2'); r2_list = append(r2_list, filename)
        data_dp = read.csv(paste0('results/', filename, '.csv'))
        column = c()
        for (d in 1:dim(data_dp)[2]){
          column = append(column, paste0("DF_", d))
        }
        colnames(data_dp) = column
        min_factors = min_n_factor_list[m]
        column = column[1:min_factors]
        data_dp = data_dp[,c(1:min_factors)]
        
        avg_ff_train_new = colMeans(data_dp[1:480,])
        data_dp$date = monthlist
        
        temp_data1 = data1
        temp_data2 = data2
  
        this_data1 = merge(temp_data1, data_dp, by="date", all.x = T)
        this_data2 = merge(temp_data2, data_dp, by="date", all.x = T)
  
        y_total = c(); y_total_test = c()
        haty_total = c(); haty_total_test = c()
        haty_capm_total = c(); haty_capm_total_test = c()
        haty_total_pred = c(); haty_total_pred_test = c()
        haty_capm_total_pred = c(); haty_capm_total_pred_test = c()
        stocks_newlist = c(); stocks_newlist_test = c()
  
        print(paste0('Start to Calculate ', filename, ' model R2'))
  
        Z_train = this_data1[, char_list]
        Z_test = this_data2[, char_list]
        R_train = this_data1[,c("xret")]
        R_test = this_data2[,c("xret")]
        stocks_newlist = append(stocks_newlist, this_data1$permno)
        stocks_newlist_test = append(stocks_newlist_test, this_data2$permno)
  
        if (g_dim_list[g] == 1){
          temp_model = append(column, "MKTRF")
          avg_ff_train_new = append(avg_ff_train_new, mean(data_ff5[1:480,1]))
        } else if (g_dim_list[g] == 5){
          temp_model = append(column, ff5)
          avg_ff_train_new = append(avg_ff_train_new, apply(data_ff5, 2, mean))
        }
  
        f_t_train = this_data1[, temp_model]
        f_t_test = this_data2[, temp_model]
  
        temp1_train = model.matrix(~as.matrix(Z_train):as.matrix(f_t_train) - 1)
        F_train = f_t_train
        F_train_inter = temp1_train
        rm("f_t_train", "temp1_train")
  
        f_t_train_pred = matrix(rep(avg_ff_train_new, dim(Z_train)[1]), nrow = dim(Z_train)[1], ncol = length(avg_ff_train_new), byrow = TRUE)
        colnames(f_t_train_pred) = temp_model
        temp1_train_pred = model.matrix(~as.matrix(Z_train):as.matrix(f_t_train_pred) - 1)
        F_train_pred = f_t_train_pred
        F_train_inter_pred = temp1_train_pred
        rm("f_t_train_pred", "temp1_train_pred")
  
        temp1_test = model.matrix(~as.matrix(Z_test):as.matrix(f_t_test) - 1)
        F_test = f_t_test
        F_test_inter = temp1_test
        rm("f_t_test", "temp1_test")
  
        f_t_test_pred = matrix(rep(avg_ff_train_new, dim(Z_test)[1]), nrow = dim(Z_test)[1], ncol = length(avg_ff_train_new), byrow = TRUE)
        colnames(f_t_test_pred) = temp_model
        temp1_test_pred = model.matrix(~as.matrix(Z_test):as.matrix(f_t_test_pred) - 1)
        F_test_pred = f_t_test_pred
        F_test_inter_pred = temp1_test_pred
        rm("f_t_test_pred", "temp1_test_pred")
  
        # R_{i,t} = a_0 + (b_0 + b^T*z_{i,t-1})*f_t + e_{i,t}
        # In-Sample Total R2 and Cross-Sectional R2
        y <- R_train; rm("R_train")
        x <- as.matrix(cbind(F_train, F_train_inter))
  
        ins_reg <- summary(lm(y ~ x))#; print(ins_reg)
        b_model <- ins_reg$coefficients[,1]
        haty <- x%*%as.matrix(b_model[2:length(b_model)])
        y_total <- append(y_total, y); rm("y")
        haty_total <- append(haty_total, haty); rm("haty")
        haty_capm_total <- append(haty_capm_total, data1[,c("MKTRF")])
        rm("x")
  
        # In-Sample Predictive R2
        x <- as.matrix(cbind(F_train_pred, F_train_inter_pred))
        haty <- x%*%as.matrix(b_model[2:length(b_model)])
        haty_total_pred <- append(haty_total_pred, haty); rm("haty")
        haty_capm_total_pred <- append(haty_capm_total_pred, rep(avg_ff_train[c("MKTRF")], dim(x)[1]))
        rm("x")
  
        # Out-of-Sample Total R2 and Cross-Sectional R2
        y <- R_test; rm("R_test")
        x <- as.matrix(cbind(F_test, F_test_inter))
        haty <- x%*%as.matrix(b_model[2:length(b_model)])
        y_total_test <- append(y_total_test, y); rm("y")
        haty_total_test <- append(haty_total_test, haty); rm("haty")
        haty_capm_total_test <- append(haty_capm_total_test, data2[,c("MKTRF")])
        rm("x")
  
        # Out-of-Sample Predictive R2
        x <- as.matrix(cbind(F_test_pred, F_test_inter_pred))
        haty <- x%*%as.matrix(b_model[2:length(b_model)])
        haty_total_pred_test <- append(haty_total_pred_test, haty); rm("haty")
        haty_capm_total_pred_test <- append(haty_capm_total_pred_test, rep(avg_ff_train[c("MKTRF")], dim(x)[1]))
        rm("x")
  
  
        # Start to output R2 results and save
        ins_TotR2 = R2.calc(y_total, haty_total, haty_capm_total)
        print(paste0(filename, " In-Sample Total R2 %: "))
        print(ins_TotR2)
        instot = append(instot, ins_TotR2)
  
        ins_CSR2 = CSR2.calc(stocks_newlist, y_total, haty_total, haty_capm_total)
        print(paste0(filename, " In-Sample Cross-Sectional R2 %: "))
        print(ins_CSR2)
        inscs = append(inscs, ins_CSR2)
  
        ins_PredR2 = R2.calc(y_total, haty_total_pred, haty_capm_total_pred)
        print(paste0(filename, " In-Sample Predictive R2 %: "))
        print(ins_PredR2)
        inspred = append(inspred, ins_PredR2)
  
        oos_TotR2 = R2.calc(y_total_test, haty_total_test, haty_capm_total_test)
        print(paste0(filename, " Out-of-Sample Total R2 %: "))
        print(oos_TotR2)
        oostot = append(oostot, oos_TotR2)
  
        oos_CSR2 = CSR2.calc(stocks_newlist_test, y_total_test, haty_total_test, haty_capm_total_test)
        print(paste0(filename, " Out-of-Sample Cross-Sectional R2 %: "))
        print(oos_CSR2)
        ooscs = append(ooscs, oos_CSR2)
  
        oos_PredR2 = R2.calc(y_total_test, haty_total_pred_test, haty_capm_total_pred_test)
        print(paste0(filename, " Out-of-Sample Predictive R2 %: "))
        print(oos_PredR2)
        oospred = append(oospred, oos_PredR2)
      }
    }
  }
}

individual_r2 = cbind(r2_list, n_layer_list, g_list, n_factor_list, regularization_list, min_n_factor, instot, inscs, inspred, oostot, ooscs, oospred)
colnames(individual_r2) = c('model_list', 'n_layer', 'g', 'n_factor', 'regularization', 'min_n_factor', 'instot','inscs','inspred','oostot','ooscs','oospred')
individual_r2 = data.frame(individual_r2)
# individual_r2 = individual_r2[(individual_r2$model_list %in% append('CAPM', finaluse_list)),]
write.csv(data.frame(individual_r2), 'individual_R2.csv')






##### Portfolio R2 #####

# Prepare for data set
rf = df_factor$RF

ff25_raw <- read.csv('../data/ff25_raw.csv')
colnames(ff25_raw)[1] <- "date"
ff25_raw <- data.frame(ff25_raw, row.names = 1)
ff25 <- ff25_raw/100 - rf

ind49_raw <- read.csv('../data/ind49_raw.csv')
colnames(ind49_raw)[1] <- "date"
ind49_raw <- data.frame(ind49_raw, row.names = 1)
ind49 <- ind49_raw/100 - rf

bisort_raw <- read.csv('../data/bisort_returns.csv')
bisort <- data.frame(bisort_raw, row.names = 1)
# bisort <- bisort_raw - rf
del_re <- c()
for (b in 1:dim(bisort)[2]){
  if (stringr::str_detect(colnames(bisort)[b], "1re") == TRUE){
    del_re = append(del_re, b)
  } else if (stringr::str_detect(colnames(bisort)[b], "2re") == TRUE){
    del_re = append(del_re, b)
  } else{
    next
  }
}
bisort <- bisort[,-del_re]
bisort <- bisort[,-dim(bisort)[2]]

bisort_3_5_raw <- read.csv('../data/bisort_returns_3_5.csv')
bisort_3_5 <- data.frame(bisort_3_5_raw, row.names = 1)
del_re <- c()
for (b in 1:dim(bisort_3_5)[2]){
  if (stringr::str_detect(colnames(bisort_3_5)[b], "1re") == TRUE){
    del_re = append(del_re, b)
  } else if (stringr::str_detect(colnames(bisort_3_5)[b], "2re") == TRUE){
    del_re = append(del_re, b)
  } else if (stringr::str_detect(colnames(bisort_3_5)[b], "3re") == TRUE){
    del_re = append(del_re, b)
  } else{
    next
  }
}
bisort_3_5 <- bisort_3_5[,-del_re]

unisort_raw <- read.csv('../data/unisort_returns.csv')
unisort <- data.frame(unisort_raw, row.names = 1)
# unisort <- unisort_raw - rf
del_re <- c()
for (u in 0:9){
  del_re = append(del_re, paste0("re",u))
}
unisort <- unisort[,-which(colnames(unisort) %in% del_re)]
unisort <- unisort[,-dim(unisort)[2]]

# Start to calculate R2
portR2.calc <- function(df_y, df_x, mat_capm, model, rtn_name){
  y_ins = df_y[1:480,]
  y_oos = df_y[481:600,]
  
  x_ins = df_x[1:480,]
  x_oos = df_x[481:600,]
  
  capm_ins = mat_capm[1:480,]
  capm_oos = mat_capm[481:600,]
  
  y_total = c(); y_total_test = c()
  haty_total = c(); haty_total_test = c()
  haty_capm_total = c(); haty_capm_total_test = c()
  haty_total_pred = c(); haty_total_pred_test = c()
  haty_capm_total_pred = c(); haty_capm_total_pred_test = c()
  stocks_newlist = c(); stocks_newlist_test = c()

  y_bar_total = c(); y_bar_total_test = c()
  beta_model = c(); beta_capm = c()
  
  for (i in 1:dim(df_y)[2]){
    stocks_newlist = append(stocks_newlist, rep(colnames(df_y)[i], dim(y_ins)[1]))
    stocks_newlist_test = append(stocks_newlist_test, rep(colnames(df_y)[i], dim(y_oos)[1]))
    
    # In-Sample Total R2 and Cross-Sectional R2
    y <- y_ins[,i]
    x <- as.matrix(x_ins)
    avg_x_train <- apply(x, 2, mean)
    
    ins_reg <- summary(lm(y ~ x))#; print(ins_reg)
    b_model <- ins_reg$coefficients[,1]
    haty <- x%*%as.matrix(b_model[2:length(b_model)])
    y_total <- append(y_total, as.numeric(y))
    haty_total <- append(haty_total, as.numeric(haty)); rm("haty")
    
    y_bar_total <- append(y_bar_total, mean(y))
    beta_model <- rbind(beta_model, b_model[2:length(b_model)])
    
    ins_capm_reg <- summary(lm(y ~ as.matrix(capm_ins)))#; print(ins_capm_reg)
    b_capm <- ins_capm_reg$coefficients[,1]
    haty_capm <- as.matrix(capm_ins)%*%as.matrix(b_capm[2:length(b_capm)])
    haty_capm_total <- append(haty_capm_total, as.numeric(haty_capm)); rm("haty_capm")
    beta_capm <- append(beta_capm, b_capm[2])
    rm("x")
    
    # In-Sample Predictive R2
    x <- matrix(rep(avg_x_train, dim(x_ins)[1]), nrow = dim(x_ins)[1], ncol = length(avg_x_train), byrow = TRUE)
    haty <- x%*%as.matrix(b_model[2:length(b_model)])
    haty_total_pred <- append(haty_total_pred, as.numeric(haty)); rm("haty")
    
    haty_capm <- as.matrix(rep(mean(capm_ins), dim(x_ins)[1]))%*%as.matrix(b_capm[2:length(b_capm)])
    haty_capm_total_pred <- append(haty_capm_total_pred, as.numeric(haty_capm)); rm("haty_capm")
    rm("x")
    
    # Out-of-Sample Total R2 and Cross-Sectional R2
    y <- y_oos[,i]
    x <- as.matrix(x_oos)
    haty <- x%*%as.matrix(b_model[2:length(b_model)])
    y_total_test <- append(y_total_test, as.numeric(y))
    haty_total_test <- append(haty_total_test, as.numeric(haty)); rm("haty")
    y_bar_total_test <- append(y_bar_total_test, mean(y))
    
    haty_capm <- as.matrix(capm_oos)%*%as.matrix(b_capm[2:length(b_capm)])
    haty_capm_total_test <- append(haty_capm_total_test, as.numeric(haty_capm)); rm("haty_capm")
    rm("x")
    
    # Out-of-Sample Predictive R2
    x <- matrix(rep(avg_x_train, dim(x_oos)[1]), nrow = dim(x_oos)[1], ncol = length(avg_x_train), byrow = TRUE)
    haty <- x%*%as.matrix(b_model[2:length(b_model)])
    haty_total_pred_test <- append(haty_total_pred_test, as.numeric(haty)); rm("haty")
    
    haty_capm <- as.matrix(rep(mean(capm_ins), dim(x_oos)[1]))%*%as.matrix(b_capm[2:length(b_capm)])
    haty_capm_total_pred_test <- append(haty_capm_total_pred_test, as.numeric(haty_capm)); rm("haty_capm")
    
  }
  
  # Calculate risk premia estimates
  reg_lambda = summary(lm(y_bar_total ~ beta_model))
  lambda_val = reg_lambda$coefficients[,1]; lambda_val = lambda_val[2:length(lambda_val)]
  haty_bar_total = as.numeric(as.matrix(beta_model)%*%as.matrix(lambda_val))
  
  reg_lambda_capm = summary(lm(y_bar_total ~ beta_capm))
  lambda_val_capm = reg_lambda_capm$coefficients[,1]; lambda_val_capm = lambda_val_capm[2]
  haty_bar_capm_total = as.numeric(lambda_val_capm*beta_capm)
  
  haty_bar_total_test = as.numeric(as.matrix(beta_model)%*%as.matrix(lambda_val))
  haty_bar_capm_total_test = as.numeric(lambda_val_capm*beta_capm)
  
  # Start to output R2 results and save
  print(paste0("Start to calculate ", rtn_name, " Portfolio R2."))
  ins_TotR2 = R2.calc(y_total, haty_total, haty_capm_total)
  print(paste0(model, " In-Sample Total R2 %: "))
  print(ins_TotR2)

  ins_CSR2 = CSR2.calc(stocks_newlist, y_total, haty_total, haty_capm_total)
  print(paste0(model, " In-Sample Cross-Sectional R2 %: "))
  print(ins_CSR2)

  ins_PredR2 = R2.calc(y_total, haty_total_pred, haty_capm_total_pred)
  print(paste0(model, " In-Sample Predictive R2 %: "))
  print(ins_PredR2)

  ins_CSR2_other = R2.calc(y_bar_total, haty_bar_total, haty_bar_capm_total)
  print(paste0(model, " In-Sample Cross-Sectional R2 % (another way): "))
  print(ins_CSR2_other)

  oos_TotR2 = R2.calc(y_total_test, haty_total_test, haty_capm_total_test)
  print(paste0(model, " Out-of-Sample Total R2 %: "))
  print(oos_TotR2)

  oos_CSR2 = CSR2.calc(stocks_newlist_test, y_total_test, haty_total_test, haty_capm_total_test)
  print(paste0(model, " Out-of-Sample Cross-Sectional R2 %: "))
  print(oos_CSR2)

  oos_PredR2 = R2.calc(y_total_test, haty_total_pred_test, haty_capm_total_pred_test)
  print(paste0(model, " Out-of-Sample Predictive R2 %: "))
  print(oos_PredR2)
  
  oos_CSR2_other = R2.calc(y_bar_total_test, haty_bar_total_test, haty_bar_capm_total_test)
  print(paste0(model, " In-Sample Cross-Sectional R2 % (another way): "))
  print(oos_CSR2_other)

  return(c(rtn_name, model, ins_TotR2, ins_CSR2, ins_PredR2, ins_CSR2_other, oos_TotR2, oos_CSR2, oos_PredR2, oos_CSR2_other))
}

port_r2 = c()

print('Start to calculate Portfolio R2: ')
g_list = c(); n_layer_list = c(); n_factor_list = c(); regularization_list = c(); min_n_factor = c()

port_rtn_list = c("FF25", "Ind49", "Bisort", "Unisort", "Bisort_3_5")
for (p in 1:length(port_rtn_list)){
  this_rtn = port_rtn_list[p]
  
  if (this_rtn == "FF25"){
    data_y = ff25
  } else if (this_rtn == "Ind49"){
    data_y = ind49
  } else if (this_rtn == "Bisort"){
    data_y = bisort
  } else if (this_rtn == "Unisort"){
    data_y = unisort
  } else if (this_rtn == "Bisort_3_5"){
    data_y = bisort_3_5
  }
  
  port_r2 = rbind(port_r2, portR2.calc(data_y, data_ff5, data_capm, "FF5", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); min_n_factor = append(min_n_factor, " ")

  port_r2 = rbind(port_r2, portR2.calc(data_y, data_ipca, data_capm, "IPCA", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); min_n_factor = append(min_n_factor, " ")
  
  port_r2 = rbind(port_r2, portR2.calc(data_y, data_rppca, data_capm, "RPPCA", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); min_n_factor = append(min_n_factor, " ")
  
  for (l in 1:length(layer_list)){
    for (g in 1:length(g_dim_list)){
      for (v in 1:length(l1_v_list)){
        for (m in 1:length(min_n_factor_list)){
          filename = paste0('factors', g_dim_list[g], '_', layer_list[l], '_5_', l1_v_list[v], '_cv2')
          data_dp = read.csv(paste0('results/', filename, '.csv'))
          column = c()
          for (d in 1:dim(data_dp)[2]){
            column = append(column, paste0("DF_", d))
          }
          colnames(data_dp) = column
          min_factors = min_n_factor_list[m]
          g_list = append(g_list, g_dim_list[g]); n_layer_list = append(n_layer_list, layer_list[l]); n_factor_list = append(n_factor_list, dim(data_dp)[2])
          regularization_list = append(regularization_list, l1_v_list[v]); min_n_factor = append(min_n_factor, min_factors)
          data_dp = data_dp[,c(1:min_factors)]
          
          if (g_dim_list[g] == 1){
            data_x = cbind(data_dp, data_capm)
          } else if (g_dim_list[g] == 5){
            data_x = cbind(data_dp, data_ff5)
          }
          port_r2 = rbind(port_r2, portR2.calc(data_y, data_x, data_capm, filename, this_rtn))
        }
      }
    }
  }
}

port_r2 = cbind(port_r2, g_list, n_layer_list, n_factor_list, regularization_list, min_n_factor)
port_r2 = data.frame(port_r2)
colnames(port_r2) = c('port','model_list','instot','inscs','inspred','inscs_other','oostot','ooscs','oospred','ooscs_other', 'g', 'n_layer', 'n_factor', 'regularization', 'min_n_factor')
port_r2 = port_r2[,c('port','model_list','g','n_layer','n_factor','regularization','min_n_factor','instot','inscs','inspred','inscs_other','oostot','ooscs','oospred','ooscs_other')]
# port_r2 = port_r2[(port_r2$model_list %in% finaluse_list),]
write.csv(data.frame(port_r2), 'Portfolio_R2.csv')



# 
# # Use IPCA Individual Beta to calculate R2
# ipca_gamma = read.csv('../data/Gamma_5_1972-01-31_19720202.csv') # 60 + 1
# colnames(ipca_gamma) = c('chars','BETA_1','BETA_2','BETA_3','BETA_4','BETA_5')
# # ipca_gamma = ipca_gamma[1:60,]
# char_list = ipca_gamma$chars[1:60]
# gamma_frame = as.matrix(ipca_gamma[,2:dim(ipca_gamma)[2]]) # 61 * 5
# beta_ipca_ins = as.matrix(cbind(data1[,char_list],1))%*%gamma_frame # n * 61 * 61 * 5 = n * 5
# haty_ipca_ins = rowSums(beta_ipca_ins*data1[,ipca])
# haty_capm_ins = as.matrix(data1$MKTRF)
# haty_ipca_ins_pred = rowSums(beta_ipca_ins*matrix(rep(apply(insaddfactor[,ipca], 2, mean), dim(data1)[1]), nrow = dim(data1)[1], ncol = length(ipca), byrow = TRUE))
# haty_capm_ins_pred = as.matrix(rep(avg_ff_train[1], dim(data1)[1]))
# 
# beta_ipca_oos = as.matrix(cbind(data2[,char_list],1))%*%gamma_frame
# haty_ipca_oos = rowSums(beta_ipca_oos*data2[,ipca])
# haty_capm_oos = as.matrix(data2$MKTRF)
# haty_ipca_oos_pred = rowSums(beta_ipca_oos*matrix(rep(apply(insaddfactor[,ipca], 2, mean), dim(data2)[1]), nrow = dim(data2)[1], ncol = length(ipca), byrow = TRUE))
# haty_capm_oos_pred = as.matrix(rep(avg_ff_train[1], dim(data2)[1]))
# 
# # Start to output R2 results and save
# print(paste0("Start to calculate iPCA Beta R2."))
# ins_TotR2 = R2.calc(as.matrix(data1$xret), haty_ipca_ins, haty_capm_ins)
# print(paste0(" In-Sample Total R2 %: "))
# print(ins_TotR2)
# 
# ins_CSR2 = CSR2.calc(data1[,c("permno")], data1[,c("xret")], haty_ipca_ins, haty_capm_ins)
# print(paste0(" In-Sample Cross-Sectional R2 %: "))
# print(ins_CSR2)
# 
# ins_PredR2 = R2.calc(as.matrix(data1$xret), haty_ipca_ins_pred, haty_capm_ins_pred)
# print(paste0(" In-Sample Predictive R2 %: "))
# print(ins_PredR2)
# 
# oos_TotR2 = R2.calc(as.matrix(data2$xret), haty_ipca_oos, haty_capm_oos)
# print(paste0(" Out-of-Sample Total R2 %: "))
# print(oos_TotR2)
# 
# oos_CSR2 = CSR2.calc(as.matrix(data2$permno), as.matrix(data2$xret), haty_ipca_oos, haty_capm_oos)
# print(paste0(" Out-of-Sample Cross-Sectional R2 %: "))
# print(oos_CSR2)
# 
# oos_PredR2 = R2.calc(as.matrix(data2$xret), haty_ipca_oos_pred, haty_capm_oos_pred)
# print(paste0(" Out-of-Sample Predictive R2 %: "))
# print(oos_PredR2)
# 
# ipca_R2 = cbind("ipca_R2", ins_TotR2, ins_CSR2, ins_PredR2, oos_TotR2, oos_CSR2, oos_PredR2)
# colnames(ipca_R2) <- c("model_list", "instot", "inscs", "inspred", "oostot", "ooscs", "oospred")
# write.csv(data.frame(ipca_R2), 'ipca_R2.csv')
# 
# 
# 
# # Use IPCA Individual Beta to calculate Portfolio Beta and R2
# chars_59 = char_list[-1]
# ins_data = cbind(data1[, c("date", "xret", "lag_me", "ffi49", char_list)], beta_ipca_ins)
# monthvec_ins = unique(ins_data$date)
# df_factor_use = cbind(df_factor[,c(1:2)], data_ipca)
# df_factor_use$date = c(df_ipca_ins$date, df_ipca_oos$date)
# 
# ff25_port = c(); ind49_port = c(); bisort354_port = c(); unisort600_port = c(); bisort885_port = c()
# for (m in 1:length(monthvec_ins)){
#   month = monthvec_ins[m]
#   print(month)
#   eachmonth = ins_data[(ins_data[,c("date")] == month), ]
#   
#   # FF25
#   break_list = seq(-1,1,0.4); label_list = c(1:5)
#   df_ff25 = eachmonth
#   df_ff25$group_me = cut(df_ff25$rank_me, breaks = break_list, include.lowest = T, labels = label_list)
#   df_ff25$group_bm = cut(df_ff25$rank_bm, breaks = break_list, include.lowest = T, labels = label_list)
#   df_ff25$group_list = paste0(df_ff25$group_me, df_ff25$group_bm)
#   port_list = unique(df_ff25$group_list)
#   port_value = c()
#   for (p in 1:length(port_list)){
#     df_port = df_ff25[(df_ff25$group_list == port_list[p]), ]
#     weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#     vw_y = as.numeric(df_port$xret%*%weight)
#     vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#     port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#     port_value = rbind(port_value, port_output)
#   }
#   ff25_port = rbind(ff25_port, port_value)
#   rm("break_list", "label_list", "port_list", "port_value")
#   
#   # Ind49
#   df_ind49 = eachmonth
#   port_list = unique(df_ind49$ffi49)
#   port_value = c()
#   for (p in 1:length(port_list)){
#     df_port = df_ind49[(df_ind49$ffi49 == port_list[p]), ]
#     weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#     vw_y = as.numeric(df_port$xret%*%weight)
#     vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#     port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#     port_value = rbind(port_value, port_output)
#   }
#   ind49_port = rbind(ind49_port, port_value)
#   rm("break_list", "label_list", "port_list", "port_value")
#   
#   # Bisort 354 (2*3*59)
#   me_break = seq(-1,1,1); me_label = c(1:2); char_break = seq(-1,1,2/3); char_label = c(1:3)
#   port_value = c()
#   for (h in 1:length(chars_59)){
#     char = str_sub(chars_59[h], 6)
#     df_bisort354 = eachmonth
#     df_bisort354$group_me = cut(df_bisort354$rank_me, breaks = me_break, include.lowest = T, labels = me_label)
#     df_bisort354$group_char = cut(df_bisort354[,c(chars_59[h])], breaks = char_break, include.lowest = T, labels = char_label)
#     df_bisort354$group_list = paste0("me", df_bisort354$group_me, "_", char, df_bisort354$group_char)
#     port_list = unique(df_bisort354$group_list)
#     for (p in 1:length(port_list)){
#       df_port = df_bisort354[(df_bisort354$group_list == port_list[p]), ]
#       weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#       vw_y = as.numeric(df_port$xret%*%weight)
#       vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#       port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#       port_value = rbind(port_value, port_output)
#     }
#   }
#   bisort354_port = rbind(bisort354_port, port_value)
#   rm("me_break", "me_label", "char_break", "char_label", "port_list", "port_value")
#   
#   # Unisort 600 (60*10)
#   break_list = seq(-1,1,0.2); label_list = c(1:10)
#   port_value = c()
#   for (h in 1:length(char_list)){
#     char = str_sub(char_list[h], 6)
#     df_unisort600 = eachmonth
#     df_unisort600$group_char = cut(df_unisort600[,c(char_list[h])], breaks = break_list, include.lowest = T, labels = label_list)
#     df_unisort600$group_list = paste0(char, df_unisort600$group_char)
#     port_list = unique(df_unisort600$group_list)
#     for (p in 1:length(port_list)){
#       df_port = df_unisort600[(df_unisort600$group_list == port_list[p]), ]
#       weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#       vw_y = as.numeric(df_port$xret%*%weight)
#       vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#       port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#       port_value = rbind(port_value, port_output)
#     }
#   }
#   unisort600_port = rbind(unisort600_port, port_value)
#   rm("break_list", "label_list", "port_list", "port_value")  
#   
#   # Bisort 885 (3*5*59)
#   me_break = c(-1,-0.4,0.4,1); me_label = c(1:3); char_break = seq(-1,1,0.4); char_label = c(1:5)
#   port_value = c()
#   for (h in 1:length(chars_59)){
#     char = str_sub(chars_59[h], 6)
#     df_bisort885 = eachmonth
#     df_bisort885$group_me = cut(df_bisort885$rank_me, breaks = me_break, include.lowest = T, labels = me_label)
#     df_bisort885$group_char = cut(df_bisort885[,c(chars_59[h])], breaks = char_break, include.lowest = T, labels = char_label)
#     df_bisort885$group_list = paste0("me", df_bisort885$group_me, "_", char, df_bisort885$group_char)
#     port_list = unique(df_bisort885$group_list)
#     for (p in 1:length(port_list)){
#       df_port = df_bisort885[(df_bisort885$group_list == port_list[p]), ]
#       weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#       vw_y = as.numeric(df_port$xret%*%weight)
#       vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#       port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#       port_value = rbind(port_value, port_output)
#     }
#   }
#   bisort885_port = rbind(bisort885_port, port_value)
#   rm("me_break", "me_label", "char_break", "char_label", "port_list", "port_value")
#   
# }
# 
# ff25_port_ins = data.frame(ff25_port); colnames(ff25_port_ins) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# ind49_port_ins = data.frame(ind49_port); colnames(ind49_port_ins) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# bisort354_port_ins = data.frame(bisort354_port); colnames(bisort354_port_ins) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# unisort600_port_ins = data.frame(unisort600_port); colnames(unisort600_port_ins) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# bisort885_port_ins = data.frame(bisort885_port); colnames(bisort885_port_ins) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# 
# ff25_port_ins = merge(ff25_port_ins, df_factor_use, by="date", all.x = T); print(dim(ff25_port_ins))
# ind49_port_ins = merge(ind49_port_ins, df_factor_use, by="date", all.x = T); print(dim(ind49_port_ins))
# bisort354_port_ins = merge(bisort354_port_ins, df_factor_use, by="date", all.x = T); print(dim(bisort354_port_ins))
# unisort600_port_ins = merge(unisort600_port_ins, df_factor_use, by="date", all.x = T); print(dim(unisort600_port_ins))
# bisort885_port_ins = merge(bisort885_port_ins, df_factor_use, by="date", all.x = T); print(dim(bisort885_port_ins))
# 
# 
# oos_data = cbind(data2[, c("date", "xret", "lag_me", "ffi49", char_list)], beta_ipca_oos)
# monthvec_oos = unique(oos_data$date)
# 
# ff25_port = c(); ind49_port = c(); bisort354_port = c(); unisort600_port = c(); bisort885_port = c()
# for (m in 1:length(monthvec_oos)){
#   month = monthvec_oos[m]
#   print(month)
#   eachmonth = oos_data[(oos_data[,c("date")] == month), ]
#   
#   # FF25
#   break_list = seq(-1,1,0.4); label_list = c(1:5)
#   df_ff25 = eachmonth
#   df_ff25$group_me = cut(df_ff25$rank_me, breaks = break_list, include.lowest = T, labels = label_list)
#   df_ff25$group_bm = cut(df_ff25$rank_bm, breaks = break_list, include.lowest = T, labels = label_list)
#   df_ff25$group_list = paste0(df_ff25$group_me, df_ff25$group_bm)
#   port_list = unique(df_ff25$group_list)
#   port_value = c()
#   for (p in 1:length(port_list)){
#     df_port = df_ff25[(df_ff25$group_list == port_list[p]), ]
#     weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#     vw_y = as.numeric(df_port$xret%*%weight)
#     vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#     port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#     port_value = rbind(port_value, port_output)
#   }
#   ff25_port = rbind(ff25_port, port_value)
#   rm("break_list", "label_list", "port_list", "port_value")
#   
#   # Ind49
#   df_ind49 = eachmonth
#   port_list = unique(df_ind49$ffi49)
#   port_value = c()
#   for (p in 1:length(port_list)){
#     df_port = df_ind49[(df_ind49$ffi49 == port_list[p]), ]
#     weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#     vw_y = as.numeric(df_port$xret%*%weight)
#     vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#     port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#     port_value = rbind(port_value, port_output)
#   }
#   ind49_port = rbind(ind49_port, port_value)
#   rm("break_list", "label_list", "port_list", "port_value")
#   
#   # Bisort 354 (2*3*59)
#   me_break = seq(-1,1,1); me_label = c(1:2); char_break = seq(-1,1,2/3); char_label = c(1:3)
#   port_value = c()
#   for (h in 1:length(chars_59)){
#     char = str_sub(chars_59[h], 6)
#     df_bisort354 = eachmonth
#     df_bisort354$group_me = cut(df_bisort354$rank_me, breaks = me_break, include.lowest = T, labels = me_label)
#     df_bisort354$group_char = cut(df_bisort354[,c(chars_59[h])], breaks = char_break, include.lowest = T, labels = char_label)
#     df_bisort354$group_list = paste0("me", df_bisort354$group_me, "_", char, df_bisort354$group_char)
#     port_list = unique(df_bisort354$group_list)
#     for (p in 1:length(port_list)){
#       df_port = df_bisort354[(df_bisort354$group_list == port_list[p]), ]
#       weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#       vw_y = as.numeric(df_port$xret%*%weight)
#       vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#       port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#       port_value = rbind(port_value, port_output)
#     }
#   }
#   bisort354_port = rbind(bisort354_port, port_value)
#   rm("me_break", "me_label", "char_break", "char_label", "port_list", "port_value")
#   
#   # Unisort 600 (60*10)
#   break_list = seq(-1,1,0.2); label_list = c(1:10)
#   port_value = c()
#   for (h in 1:length(char_list)){
#     char = str_sub(char_list[h], 6)
#     df_unisort600 = eachmonth
#     df_unisort600$group_char = cut(df_unisort600[,c(char_list[h])], breaks = break_list, include.lowest = T, labels = label_list)
#     df_unisort600$group_list = paste0(char, df_unisort600$group_char)
#     port_list = unique(df_unisort600$group_list)
#     for (p in 1:length(port_list)){
#       df_port = df_unisort600[(df_unisort600$group_list == port_list[p]), ]
#       weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#       vw_y = as.numeric(df_port$xret%*%weight)
#       vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#       port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#       port_value = rbind(port_value, port_output)
#     }
#   }
#   unisort600_port = rbind(unisort600_port, port_value)
#   rm("break_list", "label_list", "port_list", "port_value")  
#   
#   # Bisort 885 (3*5*59)
#   me_break = c(-1,-0.4,0.4,1); me_label = c(1:3); char_break = seq(-1,1,0.4); char_label = c(1:5)
#   port_value = c()
#   for (h in 1:length(chars_59)){
#     char = str_sub(chars_59[h], 6)
#     df_bisort885 = eachmonth
#     df_bisort885$group_me = cut(df_bisort885$rank_me, breaks = me_break, include.lowest = T, labels = me_label)
#     df_bisort885$group_char = cut(df_bisort885[,c(chars_59[h])], breaks = char_break, include.lowest = T, labels = char_label)
#     df_bisort885$group_list = paste0("me", df_bisort885$group_me, "_", char, df_bisort885$group_char)
#     port_list = unique(df_bisort885$group_list)
#     for (p in 1:length(port_list)){
#       df_port = df_bisort885[(df_bisort885$group_list == port_list[p]), ]
#       weight = as.matrix(df_port$lag_me/sum(df_port$lag_me))
#       vw_y = as.numeric(df_port$xret%*%weight)
#       vw_beta = as.numeric(t(weight)%*%as.matrix(df_port[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")]))
#       port_output = c(format(month, format='%Y-%m-%d'), port_list[p], vw_y, vw_beta)
#       port_value = rbind(port_value, port_output)
#     }
#   }
#   bisort885_port = rbind(bisort885_port, port_value)
#   rm("me_break", "me_label", "char_break", "char_label", "port_list", "port_value")
#   
# }
# 
# ff25_port_oos = data.frame(ff25_port); colnames(ff25_port_oos) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# ind49_port_oos = data.frame(ind49_port); colnames(ind49_port_oos) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# bisort354_port_oos = data.frame(bisort354_port); colnames(bisort354_port_oos) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# unisort600_port_oos = data.frame(unisort600_port); colnames(unisort600_port_oos) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# bisort885_port_oos = data.frame(bisort885_port); colnames(bisort885_port_oos) = c("date","port_list","vw_y","BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")
# 
# ff25_port_oos = merge(ff25_port_oos, df_factor_use, by="date", all.x = T); print(dim(ff25_port_oos))
# ind49_port_oos = merge(ind49_port_oos, df_factor_use, by="date", all.x = T); print(dim(ind49_port_oos))
# bisort354_port_oos = merge(bisort354_port_oos, df_factor_use, by="date", all.x = T); print(dim(bisort354_port_oos))
# unisort600_port_oos = merge(unisort600_port_oos, df_factor_use, by="date", all.x = T); print(dim(unisort600_port_oos))
# bisort885_port_oos = merge(bisort885_port_oos, df_factor_use, by="date", all.x = T); print(bisort885_port_oos)
# 
# portbetaR2.calc <- function(port, df_ins, df_oos){
#   y_ins = as.numeric(df_ins[,c("vw_y")])
#   beta_ipca_ins = matrix(as.numeric(unlist(df_ins[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")])), nrow = dim(df_ins)[1], ncol = 5, byrow = F)
#   haty_ipca_ins = rowSums(beta_ipca_ins*as.matrix(df_ins[,ipca]))
#   ins_capm_reg = summary(lm(y_ins ~ as.matrix(df_ins[,c("MKTRF")])))
#   haty_capm_ins = as.matrix(df_ins[,c("MKTRF")])*ins_capm_reg$coefficients[2,1]
#   haty_ipca_ins_pred = rowSums(beta_ipca_ins*matrix(rep(apply(df_factor_use[1:480,ipca], 2, mean), dim(df_ins)[1]), nrow = dim(df_ins)[1], ncol = length(ipca), byrow = TRUE))
#   haty_capm_ins_pred = as.matrix(rep(avg_ff_train[1], dim(df_ins)[1]))*ins_capm_reg$coefficients[2,1]
#   
#   y_oos = as.numeric(df_oos[,c("vw_y")])
#   beta_ipca_oos = matrix(as.numeric(unlist(df_oos[,c("BETA_1","BETA_2","BETA_3","BETA_4","BETA_5")])), nrow = dim(df_oos)[1], ncol = 5, byrow = F)
#   haty_ipca_oos = rowSums(beta_ipca_oos*as.matrix(df_oos[,ipca]))
#   haty_capm_oos = as.matrix(df_oos[,c("MKTRF")])*ins_capm_reg$coefficients[2,1]
#   haty_ipca_oos_pred = rowSums(beta_ipca_oos*matrix(rep(apply(df_factor_use[1:480,ipca], 2, mean), dim(df_oos)[1]), nrow = dim(df_oos)[1], ncol = length(ipca), byrow = TRUE))
#   haty_capm_oos_pred = as.matrix(rep(avg_ff_train[1], dim(df_oos)[1]))*ins_capm_reg$coefficients[2,1]
#   
#   print(paste0("Start to use ", port, " portfolio with iPCA port Beta to calculate R2"))
#   
#   ins_TotR2 = R2.calc(y_ins, haty_ipca_ins, haty_capm_ins)
#   print(paste0(" In-Sample Total R2 %: "))
#   print(ins_TotR2)
#   
#   ins_CSR2 = CSR2.calc(df_ins[,c("port_list")], y_ins, haty_ipca_ins, haty_capm_ins)
#   print(paste0(" In-Sample Cross-Sectional R2 %: "))
#   print(ins_CSR2)
#   
#   ins_PredR2 = R2.calc(y_ins, haty_ipca_ins_pred, haty_capm_ins_pred)
#   print(paste0(" In-Sample Predictive R2 %: "))
#   print(ins_PredR2)
#   
#   oos_TotR2 = R2.calc(y_oos, haty_ipca_oos, haty_capm_oos)
#   print(paste0(" Out-of-Sample Total R2 %: "))
#   print(oos_TotR2)
#   
#   oos_CSR2 = CSR2.calc(df_oos[,c("port_list")], y_oos, haty_ipca_oos, haty_capm_oos)
#   print(paste0(" Out-of-Sample Cross-Sectional R2 %: "))
#   print(oos_CSR2)
#   
#   oos_PredR2 = R2.calc(y_oos, haty_ipca_oos_pred, haty_capm_oos_pred)
#   print(paste0(" Out-of-Sample Predictive R2 %: "))
#   print(oos_PredR2)
#   
#   output = c(port, ins_TotR2, ins_CSR2, ins_PredR2, oos_TotR2, oos_CSR2, oos_PredR2)
#   return(output)
# }
# 
# port_ipca_beta_R2 = c()
# port_ipca_beta_R2 = rbind(port_ipca_beta_R2, portbetaR2.calc("FF25", ff25_port_ins, ff25_port_oos))
# port_ipca_beta_R2 = rbind(port_ipca_beta_R2, portbetaR2.calc("Ind49", ind49_port_ins, ind49_port_oos))
# port_ipca_beta_R2 = rbind(port_ipca_beta_R2, portbetaR2.calc("Bisort354", bisort354_port_ins, bisort354_port_oos))
# port_ipca_beta_R2 = rbind(port_ipca_beta_R2, portbetaR2.calc("Unisort600", unisort600_port_ins, unisort600_port_oos))
# port_ipca_beta_R2 = rbind(port_ipca_beta_R2, portbetaR2.calc("Bisort885", bisort885_port_ins, bisort885_port_oos))
# 
# colnames(port_ipca_beta_R2) <- c("port_list", "instot", "inscs", "inspred", "oostot", "ooscs", "oospred")
# write.csv(data.frame(port_ipca_beta_R2), 'port_ipca_beta_R2.csv')
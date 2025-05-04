library(Matrix)
library(lme4)
library(dplyr)
library(car)
library(ggplot2)

# --- Import Data ---
df <- read.csv("C:/Users/ASUS/Desktop/Thesis_Work/Dundee_data/input_to_R.csv")
print(nrow(df))
print(ncol(df))

# --- Subword Type Annotation ---
df$word_idx <- as.numeric(sapply(strsplit(df$SBWIDX, "-"), `[`, 1))
df$subword_idx <- as.numeric(sapply(strsplit(df$SBWIDX, "-"), `[`, 2))
df <- df %>%
  group_by(word_idx, TextFile) %>%
  mutate(max_subword_idx = max(subword_idx),
         subword_type = case_when(
           subword_idx == 1 ~ 'first',
           subword_idx == max_subword_idx ~ 'final',
           TRUE ~ 'middle'
         )) %>%
  ungroup()

df$subword_type <- ifelse(df$is_split == 0, 0, 
                          ifelse(df$subword_type == 'first', 1, 
                                 ifelse(df$subword_type == 'middle', 2, 
                                        ifelse(df$subword_type == 'final', 3, NA))))


# --- Frequency Transform ---
df$LogWordFreq <- log(df$WordFreq + 1)
df$LogSubwFreq <- log(df$SubwFreq)


# --- Unique Subword Type ID Assignment ---
give_unique_wordid <- function(df) {
  df$lower_SBW <- tolower(df$SBW)
  tmp_df <- data.frame(lower_SBW = unique(df$lower_SBW), word_typeid = seq_along(unique(df$lower_SBW)))
  df <- left_join(df, tmp_df, by = "lower_SBW")
  return(df)
}
df <- give_unique_wordid(df)
df <- df %>% select(-word_idx, -subword_idx, -max_subword_idx, -lower_SBW)

# --- Function: Scaling and Centering ---
preprocess_data <- function(data) {
  data <- data %>%
    mutate(
      across(c(is_split, is_punc, word_typeid, fixated, ParticipantID, TextFile, subword_type), as.factor)
    )
  
  contrasts(data$is_split) <- contr.sum(2)
  
  data <- data %>%
    mutate(across(c(SBWLEN, WordIdx, POS_Screen, Surprisal_1.0:Surprisal_10.0,
                    WordFreq, SubwFreq, LogWordFreq, LogSubwFreq),
                  ~ scale(.)[, 1], .names = "{.col}_scaled"))
  
  return(data)
}


compute_dll_for_measure <- function(df, measure_name, temperatures) {
  cat("\nProcessing reading measure:", measure_name, "\n")
  
  # Step 1: Remove zeros
  data <- df %>% filter(.data[[measure_name]] > 0)
  
  # Step 2: Remove punctuation
  data <- data %>% filter(is_punc == 0)
  
  # Step 3: Remove outliers (±3SD per Participant)
  data <- data %>%
    group_by(ParticipantID) %>%
    mutate(
      mean_val = mean(.data[[measure_name]]),
      sd_val = sd(.data[[measure_name]])
    ) %>%
    ungroup() %>%
    filter(.data[[measure_name]] >= mean_val - 3 * sd_val,
           .data[[measure_name]] <= mean_val + 3 * sd_val) %>%
    select(-mean_val, -sd_val)
  
  # Step 4: log-transformed reading measure
  data[[paste0("log_", measure_name)]] <- log(data[[measure_name]])
  
  # Step 5: Preprocess (factors & scaling)
  data <- preprocess_data(data)
  
  #print(View(data))
  print(nrow(data))
  print(ncol(data))
  
  # Step 6: Reading measure
  log_measure_col <- paste0("log_", measure_name)
  
  # Step 7: Base model
  base_model <- lmer(as.formula(paste0(log_measure_col, " ~ SBWLEN_scaled + WordIdx_scaled + POS_Screen_scaled + 
                                       LogWordFreq_scaled + LogSubwFreq_scaled + is_split +
                                       (1 | ParticipantID) + (1 | word_typeid)")),
                     data = data, REML = FALSE)
  base_loglik <- logLik(base_model) / nrow(data)
  print(summary(base_model))

  # --- Function: Per-observation delta log-likelihood ---
  delta_loglik_all <- numeric(length(temperatures))
  delta_non_split <- delta_first <- delta_middle <- delta_last <- numeric(length(temperatures))
  
  logLik_per <- function(lmer_model, data, log_col) {
    predictions <- predict(lmer_model, newdata = data, re.form = NA)
    stdev <- sigma(lmer_model)
    dnorm(data[[log_col]], mean = predictions, sd = stdev, log = TRUE)
  }
  
  # Step 8: Temperature loop
  for (i in seq_along(temperatures)) {
    temp <- temperatures[i]
    cat("\nProcessing temperature:", temp, "\n")
    surprisal_col <- paste0("Surprisal_", format(temp, nsmall = 1), "_scaled")
    
    model_formula <- as.formula(paste0(log_measure_col, " ~ SBWLEN_scaled + ", surprisal_col, " + 
                                        WordIdx_scaled + POS_Screen_scaled + 
                                        LogWordFreq_scaled + LogSubwFreq_scaled + is_split +
                                        (1 | ParticipantID) + (1 | word_typeid)"))
    
    model <- lmer(model_formula, data = data, REML = FALSE)
    target_loglik <- logLik(model) / nrow(data)
    print(summary(model))
    
    # delta-loglik (overall)
    delta_loglik_all[i] <- target_loglik - base_loglik
    
    # per-observation delta log-likelihood
    diff_ll <- logLik_per(model, data, log_measure_col) - logLik_per(base_model, data, log_measure_col)
    
    n_single <- length(unique(data$word_typeid[data$is_split == 0]))
    n_first <- length(unique(data$word_typeid[data$subword_type == 1]))
    n_middle <- length(unique(data$word_typeid[data$subword_type == 2]))
    n_last <- length(unique(data$word_typeid[data$subword_type == 3]))
    cat("n_single =", n_single, "| n_first =", n_first, "| n_middle =", n_middle, "| n_last =", n_last, "\n")
    
    delta_non_split[i] <- sum(diff_ll[data$is_split == 0], na.rm = TRUE) / max(n_single, 1)
    delta_first[i] <- sum(diff_ll[data$subword_type == 1], na.rm = TRUE) / max(n_first, 1)
    delta_middle[i] <- sum(diff_ll[data$subword_type == 2], na.rm = TRUE) / max(n_middle, 1)
    delta_last[i] <- sum(diff_ll[data$subword_type == 3], na.rm = TRUE) / max(n_last, 1)
  }
  
  # Final results
  result <- data.frame(
    Temperature = temperatures,
    Delta_LogLik_All = delta_loglik_all,
    Delta_LogLik_Non_Split = delta_non_split,
    Delta_LogLik_First = delta_first,
    Delta_LogLik_Middle = delta_middle,
    Delta_LogLik_Last = delta_last,
    N_single = n_single,
    N_first = n_first,
    N_middle = n_middle,
    N_last = n_last
  )
  
  return(list(result = result, preprocessed_data = data))
}

# --- DLL Plot (Overall) ---

plot_dll_overall <- function(dll_df, measure_name = "TRT") {
  ggplot(dll_df, aes(x = Temperature, y = Delta_LogLik_All)) +
    geom_line(size = 1.1, color = "#0072B2") +
    geom_point(size = 2, color = "#D55E00") +
    labs(
      #title = paste("Overall Delta-Log-Likelihood (", measure_name, ")", sep = ""),
      #title = paste("Dundee (gpt2-small)"),
      x = "Temperature",
      y = expression(Delta[LL])
    ) +
    theme_minimal(base_size = 20)+
    scale_x_continuous(
      breaks = seq(min(dll_df$Temperature), max(dll_df$Temperature), by = 1.5)) 
}

# --- DLL by Subword Type ---

plot_dll_by_subword_type <- function(dll_df, measure_name = "TRT") {
  require(tidyr)
  
  dll_long <- dll_df %>%
    select(Temperature,
           Single = Delta_LogLik_Non_Split,
           First = Delta_LogLik_First,
           Middle = Delta_LogLik_Middle,
           Last = Delta_LogLik_Last) %>%
    pivot_longer(cols = -Temperature, names_to = "SubwordType", values_to = "DeltaLogLik")
  
  dll_long$SubwordType <- factor(dll_long$SubwordType, levels = c("Single", "First", "Middle", "Last"))
  
  ggplot(dll_long, aes(x = Temperature, y = DeltaLogLik, color = SubwordType)) +
    geom_line(size = 1.1) +
    geom_point(size = 2) +
    labs(
      #title = paste("Delta-Log-Likelihood by Subword Type (", measure_name, ")", sep = ""),
      title = paste("Dundee (gpt2-small)"),
      x = "Temperature",
      y = expression(Delta[LL]),
      color = "Subword Type"
    ) +
    theme_minimal(base_size = 20)+
    scale_x_continuous(
      breaks = seq(min(dll_long$Temperature), max(dll_long$Temperature), by = 1.5))+ 
    theme(
      legend.position = c(1, 1), 
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = alpha("white", 0.8), color = NA),
      legend.title = element_text(face = "bold", size=13),
      legend.text = element_text(size = 12)
    )
}


# --- Mean Surprisal Plot by Subword Type ---
plot_mean_surprisal <- function(data, temperatures) {
  plot_data <- data.frame()
  
  for (temp in temperatures) {
    surprisal_col <- paste0("Surprisal_", format(temp, nsmall = 1))
    
    temp_data <- data %>%
      group_by(subword_type) %>%
      summarize(
        Mean_Surprisal = mean(.data[[surprisal_col]], na.rm = TRUE),
        SE_Surprisal = sd(.data[[surprisal_col]], na.rm = TRUE) / sqrt(n_distinct(word_typeid)),
        .groups = "drop"
      ) %>%
      mutate(Temperature = temp)
    
    plot_data <- bind_rows(plot_data, temp_data)
  }
  
  plot_data$subword_type <- factor(plot_data$subword_type,
                                   levels = c(0, 1, 2, 3),
                                   labels = c("Single", "First", "Middle", "Last"))
  
  ggplot(plot_data, aes(x = Temperature, y = Mean_Surprisal, color = subword_type)) +
    geom_line(size = 1.1) +
    geom_point(size = 2) +
    geom_ribbon(aes(ymin = Mean_Surprisal - SE_Surprisal,
                ymax = Mean_Surprisal + SE_Surprisal, fill = subword_type), alpha = 0.4, color = NA) +
    labs(title = paste("Dundee (gpt2-small)"),
      x = "Temperature", y = "Average Surprisal", color = "Subword Type", fill = "Subword Type") +
    theme_minimal(base_size = 20) +
    theme(
      legend.position = c(1, 1),
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = alpha("white", 0.8), color = NA),
      legend.title = element_text(face = "bold", size = 13),
      legend.text = element_text(size = 12)
    )
}


#temperatures <- c(1.0, 2.0, 2.5)
temperatures <- c(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                  2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0,
                  5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0)


output_dir <- "C:/Users/ASUS/Desktop/Thesis_Work/Dundee_data/"

#dll_result_TRT <- compute_dll_for_measure(df, "TRT", temperatures)
#write.csv(dll_result_TRT$result,file.path(output_dir,"delta_loglik_TRT.csv"), row.names = FALSE)
dll_result_TRT<- read.csv("C:/Users/ASUS/Desktop/Thesis_Work/Dundee_data/delta_loglik_TRT.csv")
p1 <- plot_dll_overall(dll_result_TRT, "TRT")
#p2 <- plot_dll_by_subword_type(dll_result_TRT$result, "TRT")
#p3 <- plot_mean_surprisal(dll_result_TRT$preprocessed_data, temperatures)
#ggsave(file.path(output_dir,"trt_mean_surprisal.pdf"), p3, width=8, height=6)
ggsave(file.path(output_dir,"dll_trt_overall.pdf"), p1, width=8, height=6)
#ggsave(file.path(output_dir,"dll_trt_bytype.pdf"), p2, width=8, height=6)

#dll_result_FFD <- compute_dll_for_measure(df, "FFD", temperatures)
#write.csv(dll_result_FFD$result, file.path(output_dir,"delta_loglik_FFD.csv"), row.names = FALSE)
dll_result_FFD<- read.csv("C:/Users/ASUS/Desktop/Thesis_Work/Dundee_data/delta_loglik_FFD.csv")
p1 <- plot_dll_overall(dll_result_FFD, "FFD")
#p2 <- plot_dll_by_subword_type(dll_result_FFD$result, "FFD")
#p3 <- plot_mean_surprisal(dll_result_FFD$preprocessed_data, temperatures)
#ggsave(file.path(output_dir,"ffd_mean_surprisal.pdf"), p3, width=8, height=6)
ggsave(file.path(output_dir,"dll_ffd_overall.pdf"), p1, width=8, height=6)
#ggsave(file.path(output_dir,"dll_ffd_bytype.pdf"), p2, width=8, height=6)

dll_result_FPFD <- compute_dll_for_measure(df, "FPFD", temperatures)
dll_result_FPFD<- read.csv("C:/Users/ASUS/Desktop/Thesis_Work/Dundee_data/delta_loglik_FPFD.csv")
#write.csv(dll_result_FPFD$result, file.path(output_dir,"delta_loglik_FPFD.csv"), row.names = FALSE)
#p1 <- plot_dll_overall(dll_result_FPFD$result, "FPFD")
p1 <- plot_dll_overall(dll_result_FPFD, "FPFD")
p2 <- plot_dll_by_subword_type(dll_result_FPFD, "FPFD")
p3 <- plot_mean_surprisal(dll_result_FPFD$preprocessed_data, temperatures)
ggsave(file.path(output_dir,"fpfd_mean_surprisal.pdf"), p3, width=8, height=6)
ggsave(file.path(output_dir,"dll_fpfd_overall.pdf"),p1, width=8, height=6)
ggsave(file.path(output_dir,"dll_fpfd_bytype.pdf"), p2, width=8, height=6)







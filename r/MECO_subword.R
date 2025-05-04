# --- Libraries ---
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(lme4)
library(car)

# --- Directories ---
data_dir <- "C:/Users/ASUS/Desktop/Thesis_Work/MECO_data/mgpt/meco_subword_level_lang_subj/"
#output_dir <- "C:/Users/ASUS/Desktop/Thesis_Work/MECO_data/mgpt/results/subword/"
#data_dir <- "C:/Users/ASUS/Desktop/Thesis_Work/MECO_data/gpt/gpt2/"
output_dir <- "C:/Users/ASUS/Desktop/Thesis_Work/MECO_data/gpt/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# --- Temperatures ---
T_list <- c(1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.25, 2.5, 2.75, 3, 
            3.25, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10)
#T_list <- c(1)

language_names <- c(
  "en" = "English",
  "ee" = "Estonian",
  "ge" = "German",
  "du" = "Dutch",
  "fi" = "Finnish",
  "it" = "Italian",
  "ru" = "Russian",
  "tr" = "Turkish",
  "gr" = "Greek",
  "ko" = "Korean",
  "sp" = "Spanish",
  "no" = "Norwegian"
)

# --- Unique Subword Type ID Assignment ---
give_unique_wordid <- function(df) {
  df$lower_word <- tolower(df$token)
  tmp_df <- data.frame(lower_word = unique(df$lower_word), word_typeid = seq_along(unique(df$lower_word)))
  left_join(df, tmp_df, by = "lower_word")
}

# --- Remove outliers (±3SD per Participant) ---
filter_data <- function(data, column) {
  summary <- data %>% group_by(uniform_id, scalar) %>% summarise(mean = mean(.data[[column]]), sd = sd(.data[[column]]), .groups="drop")
  data <- left_join(data, summary, by = c("uniform_id", "scalar")) %>%
    filter(between(.data[[column]], mean - 3 * sd, mean + 3 * sd)) %>%
    select(-mean, -sd)
  
  return(data)
}

# --- Function: Scaling and Centering ---
preprocess_data <- function(data) {
  data <- data %>%
    mutate(across(c(is_split, is_punc, word_typeid, uniform_id, trialid, subword_type), as.factor))
  contrasts(data$is_split) <- contr.sum(2)
  
  data <- data %>%      #, freq, log_freq
    mutate(across(c(subword_len, word_idx, sentnum, subword_num, token_rank, log_rank),
                  ~ scale(.)[, 1], .names = "{.col}_scaled")) %>%
    group_by(scalar) %>%
    mutate(surprisals_scaled = scale(surprisals)[, 1]) %>%
    ungroup()
  
  return(data)
}

# --- Function: Per-observation delta log-likelihood ---
logLik_per <- function(lmer_model, data, log_measure) {
  predictions <- predict(lmer_model, newdata = data, re.form = NA)
  dnorm(data[[log_measure]], mean = predictions, sd = sigma(lmer_model), log = TRUE)
}

# --- DLL Plot (Overall) ---
plot_overall_dll <- function(results_df, language_code, measure, output_dir) {
  p <- ggplot(results_df, aes(x = Temperature, y = Delta_LogLik)) +
    geom_line(size=1.1, color = "#0072B2") +
    geom_point(size=2, color = "#D55E00") +
    labs(
      title = paste("MECO-L1 (gpt-small)"),
      #title = language_code,
      x = "Temperature", y = expression(Delta[LL])) +
    theme_minimal(base_size = 20)+
    scale_x_continuous(
      breaks = seq(min(results_df$Temperature), max(results_df$Temperature), by = 1.5))
  ggsave(file.path(output_dir, paste0(measure, "_", language_code, "_overall.pdf")), p, width = 8, height = 6)
}

# --- DLL by Subword Type ---
plot_dll_by_subword_type <- function(results_df, language_code, measure, output_dir) {
  results_long <- results_df %>%
    select(Temperature,
           Single = Delta_LogLik_single,
           First = Delta_LogLik_first,
           Middle = Delta_LogLik_middle,
           Last = Delta_LogLik_last) %>%
    pivot_longer(cols = -Temperature, names_to = "Subword_Type", values_to = "Delta_LogLik")
  
  # Set factor levels for consistent order
  results_long$Subword_Type <- factor(results_long$Subword_Type, levels = c("Single", "First", "Middle", "Last"))
  
  p <- ggplot(results_long, aes(x = Temperature, y = Delta_LogLik, color = Subword_Type)) +
    geom_line(size = 1.1) +
    geom_point(size = 2) +
    labs(
      #title = paste("MECO-L1 (mgpt)"),
      title = language_code,
      x = "Temperature", y = expression(Delta[LL])) +
    theme_minimal(base_size = 20)+
    scale_x_continuous(
      breaks = seq(min(results_df$Temperature), max(results_df$Temperature), by = 1.5)) +
    theme(
      legend.position = c(1, 1),
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = alpha("white", 0.8), color = NA),
      legend.title = element_text(face = "bold", size=13),
      legend.text = element_text(size = 12)
    )
  
  ggsave(file.path(output_dir, paste0(measure, "_", language_code, "_subword_type.pdf")), p, width = 8, height = 6)
}

# --- Mean Surprisal Plot by Subword Type ---
plot_mean_surprisal <- function(data, temperatures, language_code, measure, output_dir) {
  plot_data <- data %>%
    group_by(scalar, subword_type) %>%
    summarize(
      Mean_Surprisal = mean(surprisals, na.rm = TRUE),
      SE_Surprisal = sd(surprisals, na.rm = TRUE) / sqrt(n_distinct(word_typeid)),
      .groups = "drop"
    ) %>%
    mutate(subword_type = factor(subword_type,
                                 levels = c(0, 1, 2, 3),
                                 labels = c("Single", "First", "Middle", "Last")))
  
  p <- ggplot(plot_data, aes(x = scalar, y = Mean_Surprisal, color = subword_type)) +
    geom_line(size = 1.1) +
    geom_point(size = 2) +
    geom_ribbon(aes(ymin = Mean_Surprisal - SE_Surprisal,
                    ymax = Mean_Surprisal + SE_Surprisal, fill = subword_type), alpha = 0.4, color = NA) +
    labs(
      title= paste("MECO-L1 (mgpt)") ,
      x = "Temperature", y = "Average Surprisal", color = "Subword Type", fill = "Subword Type") +
    theme_minimal(base_size = 16) +
    theme(
      legend.position = c(1, 1),
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = alpha("white", 0.8), color = NA),
      legend.title = element_text(face = "bold", size = 11),
      legend.text = element_text(size = 10)
    )
  
  ggsave(file.path(output_dir, paste0(measure, "_", language_code, "_mean_surprisal.pdf")), p, width = 8, height = 6)
}


compute_dll_by_measure <- function(language_code, lang_data, measure, T_list, output_dir) {
  measure_col <- measure
  cat("\nProcessing reading measure:", measure_col, "\n")
  log_measure <- paste0("log_", measure)
  
  # Step 1: Remove zeros
  lang_data <- lang_data %>% filter(.data[[measure_col]] > 0)
  #print(nrow(lang_data))
  
  # Step 2: Remove punctuation
  lang_data <- lang_data %>% filter(!is_punc)
  #print(nrow(lang_data))
  
  # Step 3: log-transformed reading measure
  lang_data[[log_measure]] <- log(lang_data[[measure_col]])
  #print(nrow(lang_data))
  
  # step 4: remove outliers
  lang_data <- filter_data(lang_data, measure_col)
  #print(nrow(lang_data))
  #print(sum(is.na(lang_data[[measure_col]])))
  #print(sum(is.na(lang_data[[log_measure]])))
  
  # Step 5: Preprocess (factors & scaling)
  lang_data <- preprocess_data(lang_data) %>% filter(!is.na(.data[[log_measure]]))
  #print(nrow(lang_data))
  
  results_df <- data.frame()
  
  for (temp in T_list) {
    temp_data <- lang_data %>% filter(scalar == temp)
    cat("Temperature:", temp, "- rows:", nrow(temp_data), "language:",language_name, "\n")
    
    print(unique(temp_data$subword_num))
    
    
    # Step 6: Base model                                #log_freq_scaled + 
    base_formula <- as.formula(paste0(log_measure, " ~ word_idx_scaled + subword_len_scaled + ",
                                      "log_freq_scaled +is_split + log_rank_scaled + (1 | word_typeid) + (1 | uniform_id)"))
    
    # Step 7: Target model                              #log_freq_scaled + 
    sup_formula <- as.formula(paste0(log_measure, " ~ surprisals_scaled + word_idx_scaled + subword_len_scaled + ",
                                     " log_freq_scaled +is_split + log_rank_scaled + ",
                                     "(1 | word_typeid) + (1 | uniform_id)"))
    
    lm_mod_base <- lmer(base_formula, data = temp_data, REML = FALSE)
    lm_mod <- lmer(sup_formula, data = temp_data, REML = FALSE)
    #print(vif(lm_mod))
    
    # delta-loglik (overall)
    DLL_avg <- as.numeric((logLik(lm_mod) - logLik(lm_mod_base)) / nrow(temp_data))
    
    # per-observation delta log-likelihood
    temp_data$delta_logLik <- logLik_per(lm_mod, temp_data, log_measure) - logLik_per(lm_mod_base, temp_data, log_measure)
    
    n_data_first <- length(unique(temp_data$word_typeid[temp_data$subword_type == 1]))
    n_data_middle <- length(unique(temp_data$word_typeid[temp_data$subword_type == 2]))
    n_data_last <- length(unique(temp_data$word_typeid[temp_data$subword_type == 3]))
    n_data_single <- length(unique(temp_data$word_typeid[temp_data$subword_type == 0]))
    
    # Aggregate and normalize delta log-likelihood by the number of data points
    delta_log_liks_single <- sum(temp_data$delta_logLik[temp_data$subword_type == 0], na.rm = TRUE) / n_data_single
    delta_log_liks_first <- sum(temp_data$delta_logLik[temp_data$subword_type == 1], na.rm = TRUE) / n_data_first
    delta_log_liks_middle <- sum(temp_data$delta_logLik[temp_data$subword_type == 2], na.rm = TRUE) / n_data_middle
    delta_log_liks_last <- sum(temp_data$delta_logLik[temp_data$subword_type == 3], na.rm = TRUE) / n_data_last
    
    
    results_df <- rbind(results_df, data.frame(
      Language = language_code,
      Temperature = temp,
      Avg_Surprisal = mean(temp_data$surprisals, na.rm = TRUE),
      Delta_LogLik = DLL_avg,
      Delta_LogLik_single = delta_log_liks_single,
      Delta_LogLik_first = delta_log_liks_first,
      Delta_LogLik_middle = delta_log_liks_middle,
      Delta_LogLik_last = delta_log_liks_last,
      Base_AIC = AIC(lm_mod_base),
      Sup_AIC = AIC(lm_mod),
      Base_BIC = BIC(lm_mod_base),
      Sup_BIC = BIC(lm_mod),
      N_Single = n_data_single, 
      N_First = n_data_first,
      N_Middle = n_data_middle,
      N_Last = n_data_last
    ))
  }
  
  write.csv(results_df, file.path(output_dir, paste0(measure, "_results_", language_code, "_subword_level.csv")), row.names = FALSE)
  plot_overall_dll(results_df, language_name, measure, output_dir)
  plot_dll_by_subword_type(results_df, language_name, measure, output_dir)
  plot_mean_surprisal(lang_data, T_list, language_name, measure, output_dir)
}

# --- Run All Languages & Measures ---
file_list <- list.files(path = data_dir, pattern = "_.*\\.csv$", full.names = TRUE)
file_list
language_codes <- unique(sub(".*_([a-z]+)_.*\\.csv$", "\\1", basename(file_list)))
#language_codes <- c('ee', 'ko')
#language_codes <- c('en')
measures <- c("fpd") # ,"total_dur" "ffd", 

for (language_code in language_codes) {
  #if (language_code %in% c('ee', 'ko')) next
  cat("\nProcessing language:", language_code, "\n")
  
  lang_files <- list.files(data_dir, pattern = paste0(".*_", language_code, ".*\\.csv$"), full.names = TRUE)
  #lang_files <- list.files(data_dir, pattern = "^subword_.*\\.en\\.csv$", full.names = TRUE)
  lang_files
  lang_data <- do.call(rbind, lapply(lang_files, function(f) {
    read_csv(f, show_col_types = FALSE) %>% mutate(source_file = basename(f))
  }))
  #View(lang_data)
  lang_data$subword_type[lang_data$is_split == 0] <- 0
  lang_data$token_rank[lang_data$token_rank == -1] <- 0
  lang_data <- lang_data %>%
    give_unique_wordid() %>%
    mutate(#log_freq = log(freq + 1),
           log_rank = log(token_rank + 1))
  
  for (m in measures) {
    cat("\nProcessing", m, "for language:", language_code, "\n")
    print(nrow(lang_data))
    language_name <- language_names[[language_code]]
    compute_dll_by_measure(language_name, lang_data, m, T_list, output_dir)
  }
}

### combine plots ############

library(readr)
library(patchwork)  

input_dir <- "C:/Users/ASUS/Desktop/Thesis_Work/MECO_data/mgpt/results/subword/"  
output_dir <- "C:/Users/ASUS/Desktop/Thesis_Work/MECO_data/mgpt/results/subword/" 

file_list <- list.files(input_dir, pattern = "fpd_results_.*\\.csv$", full.names = TRUE)
file_list

plot_from_file <- function(file_path) {
  results_df <- read_csv(file_path, show_col_types = FALSE)
  
  #language_code <- tools::file_path_sans_ext(basename(file_path))
  language_code <- sub("^fpd_results_([a-zA-Z]+)_.*\\.csv$", "\\1", basename(file_path))
  language_code
  
  
  p <- ggplot(results_df, aes(x = Temperature, y = Delta_LogLik)) +
    geom_line(color = "#0072B2") +
    geom_point(color = "#D55E00") +
    labs(title = paste(language_code), x = "Temperature", y = expression(Delta[LL])) +
    theme_minimal(base_size = 16)
  #+
   # scale_x_continuous(
  #    breaks = seq(min(results_df$Temperature), max(results_df$Temperature), by = 1.5))
  
  return(p)
}

plots <- lapply(file_list, plot_from_file)

combined_plot <- wrap_plots(plots) + plot_layout(ncol = 4) 

output_file <- paste0(output_dir, "fpd_combined_plots.pdf")
ggsave(output_file, combined_plot, width = 12, height = 8)

print(paste("Combined plot saved at:", output_file))

print(combined_plot)














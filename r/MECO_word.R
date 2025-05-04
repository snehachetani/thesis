library(dplyr)
library(readr)
library(lme4)
library(ggplot2)
library(car)
library(tidyr)


data_dir <- "C:/Users/ASUS/Desktop/Thesis_Work/MECO_data/mgpt/meco_word_level_lang_subj/"
output_dir <- "C:/Users/ASUS/Desktop/Thesis_Work/MECO_data/mgpt/results/word/"

#  Get unique language codes based on file names
file_list <- list.files(path = data_dir, pattern = ".*_.*\\.csv$", full.names = TRUE)
#file_list <- list.files(path = data_dir, pattern = ".*_ko_.*\\.csv$", full.names = TRUE)

language_codes <- unique(sub(".*_([a-z]+)_.*\\.csv$", "\\1", basename(file_list)))
#language_codes <- c('ko', 'ee')
print(language_codes)

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


T_list <- c(1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.25, 2.5, 2.75, 3, 
            3.25, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10)
#T_list <- c(10)

give_unique_wordid <- function(df) {
  df$lower_word <- tolower(df$ia)
  tmp_df <- data.frame(lower_word = unique(df$lower_word), word_typeid = seq_along(unique(df$lower_word)))
  df <- left_join(df, tmp_df, by = c("lower_word")) 
  return(df)
}

filter_data <- function(data, column) {
  summary <- data %>%
    group_by(uniform_id, scalar) %>%
    summarise(mean_value = mean(.data[[column]]), sd_value = sd(.data[[column]]),.groups="drop")
  
  filtered_data <- data %>%
    left_join(summary, by = c("uniform_id", "scalar")) %>%
    filter(.data[[column]] >= mean_value - 3 * sd_value,
           .data[[column]] <= mean_value + 3 * sd_value) %>%
    select(-mean_value, -sd_value)
  
  return(filtered_data)
}

# step4: scaling and centering
preprocess_data <- function(data) {
  data <- data %>%
    mutate(
      is_split = as.factor(is_split),
      is_punc = as.factor(is_punc),
      word_typeid = as.factor(word_typeid),
      skip = as.factor(skip),
      uniform_id = as.factor(uniform_id),
      trialid = as.factor(trialid)
    )
  
  contrasts(data$is_split) = contr.sum(2)    #overall mean
  
  data <- data %>%
    mutate_at(vars(ia_len, ianum, sentnum, freq, log_freq, subword_num),
              #mutate_at(vars(ia_len, ianum, sentnum, subword_num),
              list(scaled = ~ scale(., center = TRUE, scale = TRUE))) %>%
    group_by(scalar) %>%
    mutate(surprisal_scaled = scale(surprisal)[, 1]) %>%
    ungroup()
  
  return(data)
}


# Loop Over Each Language
for (language_code in language_codes) {
  cat("\nProcessing language:", language_code, "\n")
  if (language_code %in% c('ee', 'ko')) {
    next  # Skip this iteration
  }
  
  lang_files <- list.files(path = data_dir, pattern = paste0(".*_", language_code, "_.*\\.csv$"), full.names = TRUE)
  
  lang_data <- do.call(rbind, lapply(lang_files, function(file) {
    df <- read_csv(file, show_col_types = FALSE) %>% mutate(source_file = basename(file))
    return(df)
  }))
  
  lang_data <- lang_data %>% filter(ia != "<|endoftext|>")
  
  
  lang_data <- give_unique_wordid(lang_data)
  lang_data$log_freq <- log(lang_data$freq + 1)
  print(nrow(lang_data))
  #subdata_RT <- subset(lang_data, firstrun.dur > 0)
  subdata_RT <- subset(lang_data, firstfix.dur > 0)
  #subdata_RT <- subset(lang_data, dur > 0)
  print(nrow(subdata_RT))
  subdata_RT <- subset(subdata_RT, is_punc == FALSE)
  print(nrow(subdata_RT))
  #filtered_RT_data <- filter_data(subdata_RT, "firstrun.dur")
  #filtered_RT_data <- filter_data(subdata_RT, "dur")
  filtered_RT_data <- filter_data(subdata_RT, "firstfix.dur")
  print(nrow(filtered_RT_data))
  filtered_RT_data <- preprocess_data(filtered_RT_data)
  print(nrow(filtered_RT_data))
  
  filtered_RT_data <- filtered_RT_data %>%
    select(-skip, -singlefix.land, -fixated_ia)
  print(nrow(filtered_RT_data))
  
  #missing_rows <- filtered_RT_data[rowSums(is.na(filtered_RT_data)) > 0, ]
  #print(missing_rows)  # View rows containing NA
  print(sum(is.na(filtered_RT_data)))
  
  #filtered_RT_data$log_dur <- log(filtered_RT_data$dur)
  #filtered_RT_data$log_dur <- log(filtered_RT_data$firstrun.dur)
  filtered_RT_data$log_dur <- log(filtered_RT_data$firstfix.dur)
  
  results_df <- data.frame(
    Language = character(),
    Temperature = numeric(),
    Avg_Surprisal = numeric(),
    Delta_LogLik = numeric(),
    Base_AIC = numeric(),
    Sup_AIC = numeric(),
    Base_BIC = numeric(),
    Sup_BIC = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Loop Through Temperature List
  for (temp in T_list) {
    temp_data <- filtered_RT_data %>% filter(scalar == temp)
    language_name <- language_names[[language_code]]
    cat("Temperature:", temp, "- rows:", nrow(temp_data), "-language", language_name, "\n")         #log_freq_scaled + 
    

    base_formula <- log_dur ~ subword_num_scaled + ianum_scaled + ia_len_scaled + log_freq_scaled + 
                    (1 | word_typeid) + (1 | uniform_id)
    sup_formula <- as.formula(paste("log_dur ~ surprisal_scaled + subword_num_scaled + ianum_scaled + ia_len_scaled + 
                                 log_freq_scaled + (1 | word_typeid) + (1 | uniform_id)"))
    
    lm_mod_base <- lmer(base_formula, data = temp_data, REML = FALSE)
    lm_mod <- lmer(sup_formula, data = temp_data, REML = FALSE)
    print(vif(lm_mod))
    
    avg_surp <- mean(temp_data$surprisal, na.rm = TRUE)
    DLL_avg <- as.numeric((logLik(lm_mod) - logLik(lm_mod_base)) / nrow(temp_data))
    
    results_df <- rbind(results_df, data.frame(
      Language = language_name,
      Temperature = temp,
      Avg_Surprisal = avg_surp,
      Delta_LogLik = DLL_avg,
      Base_AIC = AIC(lm_mod_base),
      Sup_AIC = AIC(lm_mod),
      Base_BIC = BIC(lm_mod_base),
      Sup_BIC = BIC(lm_mod)
    ))

  } 
  output_file <- paste0(output_dir, "FFD_results_", language_code, "_word_level.csv")
  write.csv(results_df, output_file, row.names = FALSE)
  print(paste("Results saved at:", output_file))
  
  plot_file <- paste0(output_dir, "FFD_plot_", language_code, "_word_level.pdf")
  ggplot(results_df, aes(x = Temperature, y = Delta_LogLik)) +
    geom_line(color = "#0072B2") +
    geom_point(color = "#D55E00") +
    labs(
      title = paste(language_name),
      x = "Temperature",
      y = expression(Delta[LL])
    ) +
    scale_x_continuous(
      breaks = seq(min(results_df$Temperature), max(results_df$Temperature), by = 1.5)) +
    theme_minimal(base_size = 16)
  
  ggsave(plot_file, width = 8, height = 6)
  print(paste("Plot saved at:", plot_file))
}
 

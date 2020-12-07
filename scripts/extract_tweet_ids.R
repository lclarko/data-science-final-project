# Load Library
library(tidyverse)

# Read in CSV of tweets + tweet IDs
bcpoli <- read_csv("bcpoli.csv",
                   col_names = FALSE,
                   col_types = cols_only(X1 = col_character()))

# Keep only the tweetID Row
bcpoli_tweet_ids <- bcpoli %>% 
  rename(tweet_id = X1)

#Export Tweet IDs to csv
write_csv(bcpoli_tweet_ids, "bcpoli_tweet_ids.csv", col_names = FALSE, append = FALSE)
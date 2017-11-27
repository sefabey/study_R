library(rvest)
read_html("https://www.hamilton.edu/news/story/the-medias-effect-on-womens-body-image")
a <- read_html("https://eksisozluk.com/black-mirror--782772?p=2")
a %>% html_nodes(".info") %>% html_text()

?read_html
?searchTwitter()


twitteR::searchTwitter("#PoliSciNSF", cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl"))

out <- NULL
for (i in 1:150) {
     link <- paste0("https://eksisozluk.com/black-mirror--782772?p=",i, collapse = ",")
     a <- read_html(link)
     b <- a %>% html_nodes(".content") %>% html_text()
     out <- rbind(out,b)
}

out

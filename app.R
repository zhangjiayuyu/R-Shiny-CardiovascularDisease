################################################################################
# MGMT 590 Using R for Analytics
# Team Project #3
#
# Team #:3
######################################################################################
#load library
library(readxl)
library(caret)
library(xgboost)
library(mlbench)
library(tidyverse)
library(dplyr)
library(shinythemes)
library(shinydashboard)
###################################### Initialization #############################################
#read file
setwd("./")
dfCardio <- read.csv2("cardio_train.csv",header = T,dec=".")

# remove id column
dfCardio <- dfCardio [-c(1)]
# change the age by year
dfCardio$age<-dfCardio$age/365

dfCardio$gender[dfCardio$gender == 1] <- 0 # women
dfCardio$gender[dfCardio$gender == 2] <- 1 # men 

#remove ap_hi
dfCardio<- dfCardio[dfCardio$ap_lo<=200,]
dfCardio<- dfCardio[dfCardio$ap_lo>=30,]
#remove ap_lo
dfCardio<- dfCardio[dfCardio$ap_hi<=260,]
dfCardio<- dfCardio[dfCardio$ap_hi>=50,]

#coerce the column
colnames_df <- as.vector(colnames(dfCardio))
numeric_vars <- c("age", "height","weight", "ap_hi","ap_lo")
categorical_vars <- colnames_df[!colnames_df%in%numeric_vars]
dfCardio[categorical_vars] <- lapply(dfCardio[categorical_vars], as.factor)
dfCardio[numeric_vars] <- lapply(dfCardio[numeric_vars], as.numeric)
dfCardio_raw <- dfCardio
dfCardio_raw_num <- dfCardio[numeric_vars]
dfCardio_raw_cate <- dfCardio[categorical_vars]
#specifiy the dummy variable
dummies <- dummyVars(cardio ~ ., data = dfCardio) # create dummies for Xs
ex <- data.frame(predict(dummies, newdata = dfCardio)) # actually creates the dummies
names(ex) <- gsub("\\.", "", names(ex)) # removes dots from col names
dfCardio <- cbind(dfCardio$cardio, ex) # combine target var with Xs
names(dfCardio)[1] <- "cardio" # name target var 'y'
rm(dummies, ex) # clean environment

# Find if any linear combinations exist and which column combos they are.
# Below I add a vector of 1s at the beginning of the dataset. This helps ensure
# the same features are identified and removed.
# first save response
cardio <- dfCardio$cardio
# create a column of 1s. This will help identify all the right linear combos
dfCardio <- cbind(rep(1, nrow(dfCardio)), dfCardio[2:ncol(dfCardio)])
names(dfCardio)[1] <- "ones"
# identify the columns that are linear combos
comboInfo <- findLinearCombos(dfCardio)
# remove columns identified that led to linear combos
dfCardio <- dfCardio[, -comboInfo$remove]
# remove the "ones" column in the first column
dfCardio <- dfCardio[, c(2:ncol(dfCardio))]
# Add the target variable back to our data.frame
dfCardio <- cbind(cardio, dfCardio)
rm(cardio, comboInfo) # clean up

# data fram with only numeric variables and with only categorical variables
col_df <- as.vector(colnames(dfCardio))
num_vars <- c("age", "height","weight", "ap_hi","ap_lo")
cate_vars <- col_df[! col_df%in%num_vars]
numeric_df <- dfCardio[,num_vars]
categorical_df <- dfCardio[,cate_vars]

# remove features where the values they take on is limited
# here we make sure to keep the target variable and only those input
# features with enough variation
nzv <- nearZeroVar(dfCardio, saveMetrics = TRUE)
dfCardio <- dfCardio[, c(TRUE,!nzv$zeroVar[2:ncol(dfCardio)])]
str(dfCardio)
dfCardio[]
##################################################### shiny! ##################################################################
ui <- dashboardPage(title = 'Cardiovascular Disease Prediction Dashboard',
                    dashboardHeader(title = "Cardiovascular Disease Prediction Dashboard",titleWidth= 420),
                    dashboardSidebar(
                      sidebarMenu(
                      menuItem(
                          "Home",tabName = "home",icon = icon("dashboard")
                      ),  
                      menuItem(
                        "Descriptive analytics",tabName = "Descriptiveanalytics",icon = icon("chart-bar")
                      ),
                      menuItem(
                        "Model", tabName = "Model", icon = icon("coins")
                      ),
                      menuItem("Predict", tabName = "predict", icon = icon("heartbeat"))
                    )),
                    dashboardBody(
                      tabItems(
                        tabItem(
                          tabName = "home",
                          fluidRow(
                            valueBox("86 million", tags$p("people have at least one type of cardiovascular disease", style = "font-size: 130%;"), icon = icon("user-md"),color="red"),
                            valueBox("17.9 million", tags$p("death globally casue by Cardiovascular diseases (CVDs)", style = "font-size: 130%;"), icon = icon("skull-crossbones"),color="red"),
                            valueBox("32%", tags$p("all deaths worldwide casue by Cardiovascular diseases (CVDs)", style = "font-size: 130%;"), icon = icon("heart-broken"),color="red")
                          ),
                          fluidRow(
                            column(width=2),
                            imageOutput("image_display")
                          )
                        ),
                      tabItem(tabName = "Descriptiveanalytics",
                              fluidRow(
                                
                                box(
                                  
                                  #tags$style(HTML(".js-irs-0 .irs-single, .js-irs-0 .irs-bar-edge, .js-irs-0 .irs-bar {background: blue}")),
                                  
                                  selectInput('hist_x',"Choose a variable for histogram:", names(dfCardio_raw_num), 
                                              selected = names(dfCardio_raw_num)[[1]]),
                                  sliderInput("binCount", "Bins for histogram",
                                              min = 1, max = 10, value = 5),
                                  selectInput('bar_x', label = "Choose a variable for Bar chart:", names(dfCardio_raw_cate), 
                                              selected = names(dfCardio_raw_cate)[[1]]),
                                  selectInput('bar_fill', label = "Choose a variable for fill of Bar chart", names(dfCardio_raw_cate),
                                              selected = names(dfCardio_raw_cate)[[1]]),
                                  selectInput('box_x', label = "X axis of Box Plot:", names(dfCardio_raw_cate), 
                                              selected = names(dfCardio_raw_cate)[[1]]),
                                  selectInput('box_y', label = "Y axis of Box Plot:", names(dfCardio_raw_num), 
                                              selected = names(dfCardio_raw_num)[[1]]),
                                  selectInput('scatter_x', label = "X axis of Scatter Plot:", names(dfCardio_raw_num), 
                                              selected = names(dfCardio_raw_num)[[2]]),
                                  selectInput('scatter_y', label = "Y axis of Scatter Plot:", names(dfCardio_raw_num),
                                              selected = names(dfCardio_raw_num)[[1]]),
                                  actionButton(inputId = "click", label = "Generate Graph")
                                ),
                                box(title = "Histogram",
                                    status = "primary",
                                    solidHeader = TRUE,
                                    collapsible = TRUE,
                                    #h2("Histogram"),
                                    plotOutput("hist", height = "300px")
                                    ),
                                # box(title = "Bar chart",
                                #     status = "primary",
                                #     solidHeader = TRUE,
                                #     collapsible = TRUE,
                                #     #h2("Bar chart"),
                                #     plotOutput("bar", height = "300px")
                                # ),
                                box(title = "Bar chart fill",
                                    status = "primary",
                                    solidHeader = TRUE,
                                    collapsible = TRUE,
                                    #h2("Bar chart"),
                                    plotOutput("bar_fill", height = "300px")
                                ),
                                box(title = "Box Plot",
                                    status = "primary",
                                    solidHeader = TRUE,
                                    collapsible = TRUE,
                                    #h2("Box Plot"),
                                    plotOutput("box", height = "300px")
                                ),
                                box(title = "Scatter Plot",
                                    status = "primary",
                                    solidHeader = TRUE,
                                    collapsible = TRUE,
                                    #h2("Scatter Plot"),
                                    plotOutput("scatter", height = "300px")
                                )
                                )
                              ),
                      tabItem(tabName = "Model",
                              fluidRow(
                                box(title = "Variable estimated parameter coefficient",
                                    status = "primary",
                                    solidHeader = TRUE,
                                    collapsible = TRUE,
                                    #h2("Box Plot"),
                                    verbatimTextOutput("linear_co")
                                ),
                                box(title = "Variable coefficient p-values",
                                    status = "primary",
                                    solidHeader = TRUE,
                                    collapsible = TRUE,
                                    #h2("Box Plot"),
                                    verbatimTextOutput("linear_coP")
                                ),
                                box(title = "Summary of logistic regression",
                                    status = "primary",
                                    solidHeader = TRUE,
                                    collapsible = TRUE,
                                    #h2("Box Plot"),
                                    verbatimTextOutput("sumlm")
                                ),
                                box(title = "Model accuracy",
                                    status = "primary",
                                    solidHeader = TRUE,
                                    collapsible = TRUE,
                                    #h2("Box Plot"),
                                    verbatimTextOutput("accuracy")
                                ),
                                box(title = "Confusion matrix",
                                    status = "primary",
                                    solidHeader = TRUE,
                                    collapsible = TRUE,
                                    #h2("Box Plot"),
                                    verbatimTextOutput("confusion")
                                )
                              )
                      ),
                      tabItem(tabName = "predict",
                              fluidRow(
                                box(
                                  numericInput("age", label = "Age:", 50, min = 1, max = 150),
                                  selectInput("gender", label = "Gender(0:women,1:men):", choices = list("0" = 0, "1" = 1),selected = 0),
                                  numericInput("height", label = "Height(cm):", 160, min = , max = 250),
                                  numericInput("weight", label = "Weight(kg):", 40, min = 3, max = 600),
                                  numericInput("ap_hi", label = "Systolic blood pressure:", 140, min = 60, max = 260),
                                  numericInput("ap_lo", label = "Diastolic blood pressure:", 90, min = 30, max = 210),
                                  selectInput("cholesterol", label = "Cholesterol(1:normal,2:above normal,3:well above normal):",choices = list("1" = 1, "2" = 2, "3" = 3),selected = 1),
                                  selectInput("gluc", label = "Glucose(1:normal,2:above normal,3:well above normal):",choices = list("1" = 1, "2" = 2, "3" = 3),selected = 1),
                                  selectInput("smoke", label = "Smoke or not(0:No,1:Yes):", choices = list("0" = 0, "1" = 1),selected = 1),
                                  selectInput("alcohol", label = "Alcohol intake(0:No,1:Yes):", choices = list("0" = 0, "1" = 1),selected = 0),
                                  selectInput("active", label = "Physical activity(0:No,1:Yes):", choices = list("0" = 0, "1" = 1),selected = 0),
                                  actionButton(inputId = "pred", label = "Possibility of Cardiovascular Disease")
                                ),
                                infoBoxOutput("PredictBox", width = 6)
                            )
                        )
                    ),
                    tags$head(tags$link(rel = "stylesheet", type = "text/css", href = "custom.css"))
                    ),
                    skin = 'blue'
                    )

server <- function(input, output) {
  hist_x <- eventReactive(input$click, {
    input$hist_x
  })
  bar_x <- eventReactive(input$click, {
    input$bar_x
  })
  binCount <- eventReactive(input$click, {
    input$binCount
  })
  box_x <- eventReactive(input$click, {
    input$box_x
  })
  box_y <- eventReactive(input$click, {
    input$box_y
  })
  scatter_x <- eventReactive(input$click, {
    input$scatter_x
  })
  scatter_y <- eventReactive(input$click, {
    input$scatter_y
  })
  bar_fill_data <- eventReactive(input$click, {
    input$bar_fill
  })
  
  output$bar_fill <- renderPlot({
    ggplot(data=dfCardio_raw, aes_string(x = bar_x(), fill = bar_fill_data())) +
      geom_bar(position = "fill")+scale_fill_brewer(palette = "Blues")
  })
  
  output$image_display <- renderImage({
    list(
      src="Health1.png"
    )
  })
  
  #histogram
  output$hist <- renderPlot({
    ggplot(dfCardio_raw, aes_string(x=hist_x())) + 
      geom_histogram(color="skyblue", fill="lightblue",binwidth = binCount())
  })
  
  # #bar chart
  # output$bar <- renderPlot({
  #   ggplot(data=dfCardio_raw, aes_string(x = bar_x())) +
  #     geom_bar(color="skyblue", fill="lightblue") 
  # })
  
  #box
  output$box <- renderPlot({
    ggplot(data=dfCardio_raw, aes_string(x=box_x(),y=box_y())) + 
      geom_boxplot(color="skyblue", fill="lightblue") 
  })
  
  #scatter
  output$scatter <- renderPlot ({
    
    progress <- Progress$new()
    on.exit(progress$close())
    
    progress$set(message = 'making plot...',
                 value=0)
    n <-5
    for (i in 1:n) {
      progress$inc(1/n, detail = paste("Doing part", i))
      Sys.sleep(0.1)
    }
    ggplot(data=dfCardio_raw,aes_string(x=scatter_x(), y=scatter_y())) + 
      geom_point(color="skyblue", fill="lightblue")
  })
  
  #summary statistics table
  output$five <- renderPrint({summary(numeric_df)
  })
  
  # logit regression
  # preprocess
  # Step 1) figures out the means, standard deviations, other parameters, etc. to
  # transform each variable
  preProcValues <- preProcess(dfCardio[,2:ncol(dfCardio)], method = c("range"))
  # Step 2) the predict() function actually does the transformation using the
  # parameters identified in the previous step. Weird that it uses predict() to do
  # this, but it does!
  dfCardio_new <- predict(preProcValues, dfCardio)
  
  df <- na.omit(dfCardio_new)
  # Split the data into training and test set
  set.seed(123)
  training.samples <- df$cardio %>% 
    createDataPartition(p = 0.8, list = FALSE)
  train.data <- df[training.samples, ]
  test.data <- df[-training.samples, ]
  
  # Fit the model
  model <- glm(cardio ~., data = train.data, family = binomial)
  
  output$sumlm <- renderPrint(summary(model))
  output$linear_co <- renderPrint(summary(model)$coefficients[,1])
  output$linear_coP <- renderPrint(summary(model)$coefficients[,4])
  
  # Make predictions
  probabilities <- model %>% predict(test.data, type = "response")
  predicted.classes <- ifelse(probabilities > 0.5, "1", "0")
  # Model accuracy
  output$accuracy <- renderText(paste0(mean(predicted.classes == test.data$cardio)*100,"%"))
  # confusion matrix
  test.data$pred <- NA
  test.data$pred[probabilities >= 0.50] <- "Predicted Yes"
  test.data$pred[probabilities < 0.50] <- "Predicted No"
  output$confusion <- renderPrint(table(test.data$pred, test.data$cardio))
  
  # Predict new data
  pred_age <- eventReactive(input$pred, {
    input$age
  })
  pred_gender <- eventReactive(input$pred, {
    input$gender
  })
  pred_height <- eventReactive(input$pred, {
    input$height
  })
  pred_weight <- eventReactive(input$pred, {
    input$weight
  })
  pred_ap_hi <- eventReactive(input$pred, {
    input$ap_hi
  })
  pred_ap_lo <- eventReactive(input$pred, {
    input$ap_lo
  })
  pred_cholesterol  <- eventReactive(input$pred, {
    input$cholesterol 
  })
  pred_gluc <- eventReactive(input$pred, {
    input$gluc
  })
  pred_smoke <- eventReactive(input$pred, {
    input$smoke
  })
  pred_alco <- eventReactive(input$pred, {
    input$alcohol
  })
  pred_active <- eventReactive(input$pred, {
    input$active
  })
  
  output$PredictBox <- renderInfoBox({ 
    # preprocess input data
    gender0 <- ifelse(pred_gender() == 0, 0, 1)
    smoke0 <- ifelse(pred_smoke() == 0, 0, 1)
    alco0 <- ifelse(pred_alco() == 0, 0, 1)
    active0 <- ifelse(pred_active() == 0, 0, 1)
    
    if (pred_cholesterol()==1){
      cholesterol1 = 1
      cholesterol2 = 0
    } else if (pred_cholesterol()==2){
      cholesterol1 = 0
      cholesterol2 = 1
    } else{
      cholesterol1 = 0
      cholesterol2 = 0
    }
    
    if (pred_gluc()==1){
      gluc1 = 1
      gluc2 = 0
    } else if (pred_gluc()==2){
      gluc1 = 0
      gluc2 = 1
    } else{
      gluc1 = 0
      gluc2 = 0
    }
    age <- (pred_age() - min(dfCardio$age))/(max(dfCardio$age) - min(dfCardio$age))
    height <- (pred_height() - min(dfCardio$height))/(max(dfCardio$height) - min(dfCardio$height))
    weight <- (pred_age() - min(dfCardio$weight))/(max(dfCardio$weight)-min(dfCardio$weight))
    ap_hi <- (pred_ap_hi() - min(dfCardio$ap_hi))/(max(dfCardio$ap_hi)-min(dfCardio$ap_hi))
    ap_lo <- (pred_ap_lo() - min(dfCardio$ap_lo))/(max(dfCardio$ap_lo)-min(dfCardio$ap_lo))
    df_for_pred <- data.frame(age, gender0, height, weight,
                              ap_hi, ap_lo, cholesterol1, cholesterol2,
                              gluc1, gluc2, smoke0, alco0, active0)
    predict_prob <- predict(model, df_for_pred, type='response')
    predict_prob_hund <- round(predict(model, df_for_pred, type='response')*100,digits = 2)
    if (predict_prob >= 0.5) {
      infoBox(
        "BAD",
        paste0("The Possibility of Cardiovascular Disease is ",predict_prob_hund,"%"),
        icon = icon("alert", lib = "glyphicon"),
        color = "red",
        fill = TRUE
      )
    } else {
      infoBox(
        "GREAT",
        paste0("The Possibility of Cardiovascular Disease is ",predict_prob_hund,"%"),
        icon = icon("check", lib = 'glyphicon'),
        color = "blue", fill = TRUE
      )
    }
  })
}

shinyApp(ui = ui, server = server)


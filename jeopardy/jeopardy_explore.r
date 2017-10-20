setwd('/Users/Cam/Desktop/ds_practice/jeopardy/')
getwd()
ls()
options(prompt='R> ')
options(continue = '\t')

#save.image('jeopardy_explore')
load('jeopardy_explore')

contestants = read.csv('contestants.csv')
final_results = read.csv('final_results.csv')
locations = read.csv('locations.csv')
questions = read.csv('questions.csv')
trend = read.csv('trend.csv')


#what are we dealing with?
head(contestants)
head(final_results)
head(locations)
head(questions)
head(trend)




#This is Modal Jeopardy! we pit the most typical contestants against one another and ask the most typical jeopardy questions to see who will emerge as our typical champion


########
# contestants - who shall play our game?
########

#most frequent state

#most typical name of a jeopardy player
first_names = sort(table(contestants$player_first_name), decreasing =TRUE)
first_names[1:10]
last_names = sort(table(contestants$player_last_name), decreasing =TRUE)
last_names[1:10]

occupations = sort(table(contestants$occupation), decreasing =TRUE)
occupations[1:10]
#note the last occupation 'the reigning NFL MVP'
#Matt Jennings is the most common player name on jeopardy!
#Ken Jennings == GOAT confirmed, he was 74 of those 86 games
# we need someone different that old Ken Jennings, so lets take the unique ids and find our three most typical players


#redo with only unique player ids

#duplicate players removed
head(contestants)
unique_contestants = contestants[!duplicated(contestants$player_id),]
head(unique_contestants)

#are all the individuals retained?
length(unique(contestants$player_id)) == length(unique_contestants$player_id)


#who is the most typical jeopardy contestant?
u_first_name = sort(table(unique_contestants$player_first_name), decreasing =TRUE)
u_first_name[1:3]
u_last_name = sort(table(unique_contestants$player_last_name), decreasing =TRUE)
u_last_name[1:3]

u_occupations = sort(table(unique_contestants$occupation), decreasing =TRUE)
u_occupations[1:3]

u_locations = sort(table(unique_contestants$hometown_city), decreasing =TRUE)
u_locations[1:3]

#the modal 
#So who do we have playing our game of modal jeopardy this evening?

#chair 1. John Johnson # an attorney from Los Angeles
#chair 2. Matt Brown # a senior (college senior in USA is 4th year student, not an old person) from New York
#chair 3. Michael Smith # a graduate student from Washington


#with our new contestants, lets being the contest. Alex would you kindly tell us the categories

#########
# questions - what will the categories be?
#########
head(questions)

#is potent potables really the most used category?
category_freq = sort(table(questions$category), decreasing=TRUE)
#remove final jeopardy from the list

category_freq = category_freq[2:length(category_freq)]
category_freq[1:6]
length(category_freq)
#32000 different categories!
category_freq[1:10]
#most common is before and after with 468

barplot(category_freq[1:101])
barplot(category_freq)

#Our categories, first round:
category_freq[1:6]
#BEFORE & AFTER       
#POTPOURRI         
#SCIENCE 
#AMERICAN HISTORY   
#STUPID ANSWERS
#WORD ORIGINS       


#Our categories, second round:
category_freq[7:12]
#RHYME TYME
#POP CULTURE
#NONFICTION
#LITERATURE
#AMERICANA
#COLLEGES & UNIVERSITIES



#John Johnson has is our reigning modal champion, so he selects first.


#find the most typical board slections for each clue_picker in the trend df

left_selections = trend[trend$clue_picker == "returning_champ",]
middle_selections = trend[trend$clue_picker == "middle",]
right_selections = trend[trend$clue_picker == "right",]

left_choice_mat  = table(left_selections$row,left_selections$column)
left_choice_mat[] = rank(left_choice_mat)

middle_choice_mat  = table(middle_selections$row,middle_selections$column)
middle_choice_mat[] = rank(middle_choice_mat)

right_choice_mat  = table(right_selections$row,right_selections$column)
right_choice_mat[] = rank(right_choice_mat)

#use this to select the cell to pick, turn to NA once picked
right_choice_mat  == min(right_choice_mat)

#find the most typical correct respondant for each cell on the board in the trend column, use these to get the game scored.
#call a cell, and pick the winner from the list of correct respondants for that cell.
correct_respondant_mat = matrix(data = rep(0, times = 30),5,6)
for(i in 1:5){
	for(j in 1:6){
		selected_row = trend[trend$row == i,]
		selected_cell = selected_row[selected_row$column == j,]
		selected_weighted_prob = sample(selected_cell$correct_respondent,1)
		correct_respondant_mat[i,j] = as.character(selected_weighted_prob)
	}
}


correct_respondant_mat



#build a matrix with the 6 most asked questions for each category, and second matrix with the correct respondant, award the points accordingly

question_board = data.frame(BEFORE_and_AFTER = rep(0, times=6), 
							POTPOURRI = rep(0, times=6),
							SCIENCE = rep(0, times=6),
							AMERICAN_HISTORY  =rep(0, times=6),
							STUPID_ANSWERS = rep(0, times=6),
							WORD_ORIGINS = rep(0, times=6))

answer_board = question_board

#pull 6 random questions from that category and add them to the question df
for(col in 1:6){
	name = category_freq[col]
	q_df = questions[questions$category == labels(name),]
	q_to_pick = sample(1:length(q_df$answer),6)
	linear_questions = q_df$question[q_to_pick]
	answers = q_df$answer[q_to_pick]
	question_board[,col] = linear_questions
	answer_board[,col] = answers
}


	




#build a matrix with the visited cells, and use this for question selection
visited_cells = matrix(data = rep(0, times = 30),5,6)


#walk through the game, giving the points based on the correct answer modes.


#write a function to walk from question to question and then give the points to the players. 

























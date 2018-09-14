import numpy as np

# returns sizes of each story, length of each line, np version of story
# also returns the same for queries
def from_batch(batch):
	story_tokens = []
	story_sizes = []
	story_sentence_lengths = []
	query_tokens = []
	query_sentence_lengths = []
	answers = []
	for line in batch:
		# 0. divide each line into a story, a query, and an answer
		story, query, answer = line.split("::")

		# 1. get from a story
		# 1.1. divide a story into individual lines
		story_lines = story.split('\t')
		story_sizes.append(len(story_lines))
		# 1.2. divide each line into tokens
		story_lines = [x.split(' ') for x in story_lines] # list of lists
		story_tokens.extend(story_lines) # story_tokens takes all lines within a batch
		story_sentence_lengths.extend([len(l) for l in story_lines])

		# 2. get from a query
		query = query.split(' ')
		query_tokens.append(query)
		query_sentence_lengths.append(len(query))

		# 3. get answers
		answers.append(int(answer.strip()))

	# 4. get max sequence lengths for stories and queries
	max_story_length = max(story_sentence_lengths)
	max_query_length = max(query_sentence_lengths)

	# 5. create matrices
	story_inputs = np.zeros([len(story_sentence_lengths),max_story_length],dtype=int)
	query_inputs = np.zeros([len(query_sentence_lengths),max_query_length],dtype=int)
	for i, s_line in enumerate(story_tokens):
		story_inputs[i,0:len(s_line)]+=np.array(s_line,dtype=int)
	for i, q_line in enumerate(query_tokens):
		query_inputs[i,0:len(q_line)]+=np.array(q_line,dtype=int)

	# pack them into tuples
	S = (story_inputs, story_sentence_lengths,story_sizes)
	Q = (query_inputs, query_sentence_lengths)
	A = np.array(answers)

	return S,Q,A
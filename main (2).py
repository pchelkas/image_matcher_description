
# token1 = nlp(u'bird')
# token2 = nlp(u'eagle')
# print (token1.similarity(token2))

# doc = nlp(u'Andrew took many pictures with it')

# noun_chunks = list(doc.noun_chunks)
# print(noun_chunks)

# for i in range(25):
# 	f = open('./dataset-upd/' + str(i + 1) + '/input')
# 	doc = nlp(f.read())

# 	for token in doc:
# 	    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
# 	          token.shape_, token.is_alpha, token.is_stop)
# 	f.close()
# 	print()

# f = open('./dataset/1/input')

# print (f.read())

# print (token1.similarity(token2))





# --

import spacy
import requests
# If you are using a Jupyter notebook, uncomment the following line.
#%matplotlib inline
from PIL import Image
from io import BytesIO
import time
import json
from nlp import *
from threading import Thread

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
        
def getProcessedString(input):
	doc = nlp(input)
	input_sent = ''
	for token in doc:
		if token.lemma_ != 'be' and (token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ'):
			if len(input_sent) != 0:
				input_sent += ' '
			input_sent += token.lemma_
	return input_sent

fullTest = [6, 6, 3, 4, 1, 3, 1, 3, 3, 1, 2, 4, 3, 3, 1, 6, 3, 5, 2, 3, 1, 3, 1, 4, 2]

def image_caption(startFolder, test):
	acc = 0.0

	nlp = spacy.load('en_core_web_sm')

	# Replace <Subscription Key> with your valid subscription key.
	counter = 0
	a = time.time()
	subscription_key = ["990caaa48d294c5881971c4e01171630", "ed0475388c2343038f7e67935bb7d6d3"]
	assert subscription_key

	vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

	analyze_url = vision_base_url + "analyze"
	cases = 0
	# Set image_path to the local path of an image that you want to analyze.
	for folder in range(startFolder, startFolder+3):
		print ('FOLDER', folder)
		cases += 1
		f = open('./test2/'+str(folder)+'/input')
		input = f.read()
		f.close()
#		input_sent = getProcessedString(input)
				
		doc = nlp(input)
		input_sent = ''
		for token in doc:
			if token.lemma_ != 'be' and (token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ'):
				if len(input_sent) != 0:
					input_sent += ' '
				input_sent += token.lemma_
		
		
		
		# print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
		#       token.shape_, token.is_alpha, token.is_stop)
		print (input_sent)
		max_conf = -1
		max_pic = 0
		for number in range(1,2):	
			image_path = "./test2/"+str(folder)+"/"+str(number)+".jpg"
			image_data = open(image_path, "rb").read()
			headers    = {'Ocp-Apim-Subscription-Key': subscription_key[counter%2],
					  'Content-Type': 'application/octet-stream'}
			params     = {'visualFeatures': 'Tags,Categories,Description,Color'}
			response = requests.post(
				analyze_url, headers=headers, params=params, data=image_data)
			response.raise_for_status()

			# The 'analysis' object contains various fields that describe the image. The most
			# relevant caption for the image is obtained from the 'description' property.
			analysis = response.json()
			tags_sent = ''
			
			for item in analysis['tags']:
				# if i < 4 and item['name'] != 'indoor' and item['name'] != 'outdoor':
				if item['confidence'] > 0.3 and item['name'] != 'indoor' and item['name'] != 'outdoor':
						tags_sent += item['name'] + ' '
			counter = counter + 1 
			if len(tags_sent) > 0:
			#	tags_sent = getProcessedString(tags_sent)
				doc = nlp(tags_sent)
				tags_s = ''
				for token in doc:
					if token.lemma_ != 'be' and (token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ'):
						if len(tags_s) != 0:
							tags_s += ' '
						tags_s += token.lemma_
			
				sim = nlp(input_sent).similarity(nlp(tags_s))
	#			sim = similarity(input_sent, tags_s, True)

				if sim > max_conf:
					max_conf = sim
					max_pic = number
				print (tags_s, '|', sim , end='\n\n')
			else:
				print ('<no tags with such confidence found>', end='\n\n')
		# print ('MAX AT:', max_pic, end='\n\n')
		print("folder", folder)
		print("startfolder", startFolder)
		print("test value", str(test[folder - startFolder]))
		if max_pic == test[folder - startFolder]:
			
			acc += 1
		print ('max:', max_pic, '| correct:', test[folder - startFolder])

		f = open('./test2/'+str(folder)+'/output', 'w')
		f.write(str(max_pic))
	b = time.time()
	print ('accuracy:', str(acc / cases))
	#print(b - a)
	#print(counter)
	#return str(acc / cases)
	
	
twrv1 = ThreadWithReturnValue(target=image_caption, args=(1, fullTest[0:3]))
twrv2 = ThreadWithReturnValue(target=image_caption, args=(4, fullTest[3:6]))
twrv3 = ThreadWithReturnValue(target=image_caption, args=(7, fullTest[6:9]))
twrv4 = ThreadWithReturnValue(target=image_caption, args=(10,fullTest[9:12]))

twrv5 = ThreadWithReturnValue(target=image_caption, args=(13,fullTest[12:15]))
twrv6 = ThreadWithReturnValue(target=image_caption, args=(16,fullTest[15:18]))
#twrv7 = ThreadWithReturnValue(target=image_caption, args=(19,))
#twrv8 = ThreadWithReturnValue(target=image_caption, args=(22,))

twrv1.start()
twrv2.start()
#twrv3.start()
#twrv4.start()
#twrv5.start()
#wrv6.start()
#twrv7.start()
#twrv8.start()



print (twrv1.join())
print (twrv2.join())
#print (twrv3.join())
#print (twrv4.join())
#print (twrv5.join())
#print (twrv6.join())
#print (twrv7.join())
#print (twrv8.join())


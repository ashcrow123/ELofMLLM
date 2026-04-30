SPEAKER_GENERATION='''
**TASK:**
You are a language learner. You are studying a special artificial language that has !<INPUT 0>! letters. 
You can see them in the **LETTERS** section. Your task is to generate a word to express **TARGET IMAGE**, which will be sent 
to other language learner for him to guess the current **TARGET IMAGE**. You will be given your own 
**LEARNED VOCABULARY**, which can help you generate correct words.  

**LETTERS:**
!<INPUT 1>!

**REQUIREMENTS:**

1. You must generate exactly one word in this artificial language to describe the current object, using only 
the **LETTERS** listed.

2. You may reuse the same letter in the word, But the total number of letters used (including duplicates) must 
be between 1 and !<INPUT 2>!.

3. The information in the **LEARNED VOCABULARY** may be crucial to forming your new word. Please reference it carefully and thoughtfully.

4.For your convenience in observation and decision-making, all letters in words provided to you are separated by a "-".To ensure the 
standardization of the output, use the symbol '-' to space the letters in the word.
For example, <letter_1>-<letter_2>-<letter_3>-<letter_4>

5.Before you output the word, do a brief analysis of the words you want to output to help you output more suitable word.

6.Output analysis and word in json format, like **EXPECTED FORMAT**, please do not output any additional content.

**EXPECTED FORMAT:**
{
    "analysis": <do a brief analysis of the words you want to output according to **LEARNED VOCABULARY**,**OBJECT PROPERTIES** and **FAILED COMMUNICATION RECORDS**>,
    "word": <Output only the final word, without any additional information or characters,ensuring that letters are separated by a '-'>
}
'''

LISTENER_SELECTION='''
**TASK:**
You are a language learner. You are studying a special artificial language that has !<INPUT 0>! letters. You can see them in the
**LETTERS** section. Your task is to, upon receiving a **WORD** in the current artificial language, select the most likely image from **CHOICE IMAGES** that the word refers to, based on **LEARNED VOCABULARY**.

**LETTERS:**
!<INPUT 1>!

**REQUIREMENTS:**

1. **CHOICE IMAGES** are labeled with [A], [B], [C], [D], [E]. You need to select the image that best matches the meaning of the given **WORD**.

2. In the JSON output, the "option" field must contain only one key (A, B, C, D, or E) of the selected image.

3. The information in the **LEARNED VOCABULARY** maps images to words in this artificial language. Use it to understand how words relate to images.

4. Before you output your answer, do a brief analysis of the word and the choice images to help you make a better selection.

5. Output in json format, like **EXPECTED FORMAT**, please do not output any additional content.

6. For your convenience in observation and decision-making, all letters in words provided to you are separated by a "-".

**EXPECTED FORMAT:**
{
    "analysis": <brief analysis of the word meaning and how it relates to the choice images based on LEARNED VOCABULARY>,
    "option": <Directly return one option from "A","B","C","D","E">
}
'''

SPEAKER_RETRIEVAL='''
**TASK**
Your task is to select images from **ALL IMAGES** that you believe are similar in meaning or closely related to the **GIVEN IMAGE**.

**REQUIREMENTS:**

1.Each image in **ALL IMAGES** has a corresponding serial number. If you think the image is qualified, put its corresponding serial number into the final output.

2.Please provide an analysis text for this task before outputting the serial number list.

3.Strictly output according to the format, please do not output any content outside the list.

4.Here is the format we would like you to return(in **EXPECTED FORMAT**)

5.num_list must contain at most one serial number: the single image that you think is most similar in meaning or closely related to the **GIVEN IMAGE**.
If you don't think there are any pictures that meet the requirements, you can output an empty list.

**EXPECTED FORMAT:**

{
    "analysis":"<Provide your analysis of the current task>",
    "num_list":[<zero or one serial number of the image that you think is most similar in meaning or closely related to the **GIVEN IMAGE**>]
}
'''

LISTENER_RETRIEVAL='''
**TASK:**
    You are a language learner. You are studying a special artificial language that has !<INPUT 0>! letters.  You can see them in the 
    **LETTERS** section.  
    Your task is to select words from the **LEARNED VOCABULARY** that you think are similar in meaning or closely related to the **GIVEN WORD**.

**LETTERS:**
!<INPUT 1>!

**LEARNED VOCABULARY:**
!<INPUT 2>!

**GIVEN WORD:**
!<INPUT 3>!

**REQUIREMENTS:**
1. Please output the result strictly as a JSON object that follows **EXPECTED FORMAT**.

2.The information in the **LEARNED VOCABULARY** may be crucial for your selecting. Please reference it carefully and thoughtfully.

3.For your convenience in observation and decision-making, all letters in words provided to you are separated by a "-".

4.There will be a numerical number before each word in **LEARNED VOCABULARY**. When you want to choose this word, simply give its serial number.


**EXPECTED FORMAT:**

{
    "analysis":<Provide your analysis based on **LEARNED VOCABULARY** and **GIVEN WORD**>,
    "num_list":<Give the numerical serial numbers of words that you think are similar in meaning or closely related to the **GIVEN WORD**>
}

'''

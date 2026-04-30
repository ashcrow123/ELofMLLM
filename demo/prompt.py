SPEAKER_PROMPT='''
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

LISTENER_PROMPT='''
**TASK:**
You are a language learner. You are studying a special artificial language that has !<INPUT 0>! letters. You can see them in the
**LETTERS** section. Your task is to, upon receiving a **WORD** in the current artificial language, select the most likely image from **CHOICE IMAGES** that the word refers to, based on **LEARNED VOCABULARY**.

**LETTERS:**
!<INPUT 1>!

**REQUIREMENTS:**

1. **CHOICE IMAGES** are labeled with [A], [B], [C], [D], [E]. You need to select the image that best matches the meaning of the given **WORD**.

2. Please output only a single string that contains the key (A, B, C, D, or E) of the selected image—do not include any additional characters or information.

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

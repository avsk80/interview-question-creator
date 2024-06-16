from src.helper2 import llm_pipeline

# # CASE 1
# answer_gen_chain, ques_list = llm_pipeline(file_path="/home/learning/gen_ai/one_neuron/interview-question-creator/static/docs/Linux.pdf")

# # CASE 2
# # answer_gen_chain, ques_list = llm_pipeline(file_path="/home/learning/gen_ai/one_neuron/interview-question-creator/static/docs/Linux.pdf", txt="/home/learning/gen_ai/one_neuron/interview-question-creator/static/docs/linux_ques.txt")

# print(ques_list)

# for ques in ques_list:
#     print(ques)

# print("-----------------------------------")

# print(ques_list[0])

# ans_0 = answer_gen_chain.invoke(ques_list[0])

# print(ans_0['result'])

import os
import csv

def generate_ans_from_csv(pdf_filename, csv_filename=None):
    # Assuming your llm_pipeline can handle CSV and PDF input together
    answer_generation_chain, ques_list = llm_pipeline(pdf_filename, csv_filename)
    print(ques_list)
    print(answer_generation_chain)
    print(type(answer_generation_chain))
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = os.path.join(base_folder, "QA.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer", "page_no"])  # Writing the header row

        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain(question)
            print("Answer: ", answer['result'])

            # print(answer.keys())
            # print(len(answer['source_documents']))
            # print(type(answer['source_documents'][0]))
            # print(answer['source_documents'][0])
            # print(answer['source_documents'][1])
            page_no = [doc.metadata['page'] for doc in answer['source_documents']]
            print("Page numbers:", str(page_no))
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer, page_no])
    # return output_file

generate_ans_from_csv(pdf_filename="/home/learning/gen_ai/one_neuron/interview-question-creator/static/docs/Linux.pdf", csv_filename="/home/learning/gen_ai/one_neuron/interview-question-creator/static/docs/linux_ques.txt")
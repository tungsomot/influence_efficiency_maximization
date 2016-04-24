# coding = utf-8
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python cited_graph_processor.py input_file_name output_file_name"
        exit(1)
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    # print input_file_name,output_file_name
    with open(output_file_name,'w') as file_output:
        with open(input_file_name,'r') as file_input:
            line = file_input.readline().strip()
            while (line != None and line != ''):
                v, u = line.split()
                edge = u + '\t' + v +'\n'
                file_output.write(edge)
                line = file_input.readline().strip()
                # print v,u
        file_input.close()
    file_output.close()

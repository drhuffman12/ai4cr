
def data_stats(file_path)
    contents = File.read(file_path)
    # puts "contents.class: #{contents.class}"
    data = contents.each_char.to_a.sort
    puts "data.class: #{data.class}"

    # hist = Hash[*data.group_by{ |v| v }.flat_map{ |k, v| [k, v.size] }]
    hist = data.group_by{ |v| v }.map{ |k, v| [k, {char_code: k.ord, char_chr: k.ord.chr, char_count: v.size, char_ascii: k.ascii?}] }.to_h
    # puts "hist: #{hist.pretty_inspect}"

    puts "file_path: #{file_path}"
    puts "data.size: #{data.size}"
    # puts "hist[hist.keys.first][:char_code]: #{hist[hist.keys.first][:char_code]}"
    # puts "hist[hist.keys.last][:char_code]: #{hist[hist.keys.last][:char_code]}"
    puts "hist[hist.keys.first]: #{hist[hist.keys.first]}"
    puts "hist[hist.keys.last]: #{hist[hist.keys.last]}"
    puts "hist.keys.size: #{hist.keys.size}"
    puts ""
end

file_path = "spec_examples/support/neural_network/data/shakespear.txt"
data_stats(file_path)

file_path = "spec_examples/support/neural_network/data/wiki.txt"
data_stats(file_path)

file_path = "spec_examples/support/neural_network/data/linux.txt"
data_stats(file_path)

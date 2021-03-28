require "./../src/ai4cr"

class Runner
  getter file_path : String

  def initialize(@file_path)
  end
  
  def compare_successive_training_rounds(
    io_offset, time_col_qty,
    inputs_sequence, outputs_sequence,
    hidden_layer_qty, hidden_size_given,
    qty_new_members,
    my_breed_manager, max_members,
    train_qty,
    io_set_text_file
  )
    # it "successive generations score better (i.e.: lower errors)" do
    # TODO: (a) move to 'spec_bench' and (b) replace here with more 'always' tests
  
    puts
    puts "v"*40
    puts "successive generations score better (?) .. max_members: #{max_members} .. start"
    when_before = Time.local
    puts "when_before: #{when_before}"
    puts "file_path: #{file_path}"
    puts
  
    params = Ai4cr::NeuralNetwork::Rnn::RnnSimple.new(
      io_offset: io_offset,
      time_col_qty: time_col_qty,
      input_size: inputs_sequence.first.first.size,
      output_size: outputs_sequence.first.first.size,
      hidden_layer_qty: hidden_layer_qty,
      hidden_size_given: hidden_size_given
    ).config
  
    # puts
    # puts "first_gen_members: #{first_gen_members}"
    puts "inputs_sequence.size: #{inputs_sequence.size}"
    puts "inputs_sequence.first.size: #{inputs_sequence.first.size}"
    puts "inputs_sequence.first.first.size: #{inputs_sequence.first.first.size}"
    puts "inputs_sequence.class: #{inputs_sequence.class}"
    puts "outputs_sequence.class: #{outputs_sequence.class}"
    puts "params: #{params}"
  
    puts "* build/train teams"
    puts "\n  * first_gen_members (building)..."
    first_gen_members = my_breed_manager.build_team(qty_new_members, **params)
    puts "\n  * second_gen_members (breeding and training; after training first_gen_members)..."
    second_gen_members = my_breed_manager.train_team_using_sequence(inputs_sequence, outputs_sequence, first_gen_members, io_set_text_file, max_members, train_qty) # , block_logger: train_team_using_sequence_logger) 
    puts "\n  * third_gen_members (breeding and training; after training second_gen_members) ..."
    third_gen_members = my_breed_manager.train_team_using_sequence(inputs_sequence, outputs_sequence, second_gen_members, io_set_text_file, max_members, train_qty) # , block_logger: train_team_using_sequence_logger)
  
    puts "* score and stats ..."
    # puts "  * first_gen_members ..."
    p "."
    first_gen_members_scored = first_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
    first_gen_members_stats = first_gen_members.map { |member| member.error_hist_stats }
  
    # puts "  * second_gen_members ..."
    p "."
    second_gen_members_scored = second_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
    second_gen_members_stats = second_gen_members.map { |member| member.error_hist_stats }
  
    # puts "  * third_gen_members ..."
    p "."
    third_gen_members_scored = third_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
    third_gen_members_stats = third_gen_members.map { |member| member.error_hist_stats }
  
    puts
    puts "#train_team_using_sequence (text from Bible):"
    puts
    puts "first_gen_members_scored: #{first_gen_members_scored}"
    first_gen_members_stats.each { |m| puts m }
  
    puts
    puts "second_gen_members_scored: #{second_gen_members_scored}"
    second_gen_members_stats.each { |m| puts m }
  
    puts
    puts "third_gen_members_scored: #{third_gen_members_scored}"
    third_gen_members_stats.each { |m| puts m }
  
    when_after = Time.local
    puts "when_after: #{when_after}"
    when_delta = when_after - when_before
    puts "when_delta: #{(when_delta.total_seconds / 60.0).round(1)} minutes
      "
    puts
    puts "successive generations score better (?) .. max_members: #{max_members} .. end"
    puts "-"*40
    puts
  
    # expect(second_gen_members_scored).to be < first_gen_members_scored
    # expect(third_gen_members_scored).to be < second_gen_members_scored
  
    # end
    # rescue e
    #   # puts "e:"
    #   # puts "  class: #{e.class}"
    #   # puts "  message: #{e.message}"
    #   # puts "  backtrace: #{e.backtrace}"
    #   raise e
    # ensure
    #   when_after = Time.local
    #   puts "when_after: #{when_after}"
    #   when_delta = when_after - when_before
    #   puts "when_delta: #{when_delta.total_seconds / 60.0} minutes
    #   "
    #   puts
    #   puts "successive generations score better (?) .. max_members: #{max_members} .. end"
    #   puts "-"*40
    #   puts
  end  
end

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

my_breed_manager = Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager.new


file_path = "./spec_bench/support/neural_network/data/bible_utf/eng-web_002_GEN_01_read.txt"
file_type_raw = Ai4cr::Utils::IoData::FileType::Raw
prefix_raw_qty = 0
prefix_raw_char = " "
chars_at_a_time = 0
  
io_set_text_file = Ai4cr::Utils::IoData::TextFileIodBits.new(
# io_set_text_file = Ai4cr::Utils::IoData::TextFileIodFloat.new(
  file_path, file_type_raw,
  prefix_raw_qty, prefix_raw_char,
  chars_at_a_time
)

# re 'compare_successive_training_rounds'
time_col_qty = 6 # 25
io_offset = time_col_qty
# ios = io_set_text_file.iod_to_io_set_with_offset_time_cols(time_col_qty, io_offset)
ios = io_set_text_file.iod_to_io_set_with_offset_time_cols(time_col_qty, io_offset)

inputs_sequence = ios[:input_set]
outputs_sequence = ios[:output_set]

hidden_layer_qty = 3
hidden_size_given = 100 # 100 # 200

max_members = 10
qty_new_members = max_members

train_qty = 4

puts
puts "*"*40
puts "my_breed_manager: #{my_breed_manager}"
puts "io_set_text_file: #{io_set_text_file}"
puts "v"*40
puts "io_set_text_file.raw: #{io_set_text_file.raw}"
puts "^"*40
puts
puts "io_set_text_file.raw.size: #{io_set_text_file.raw.size}"
puts "io_set_text_file.raw.size: #{io_set_text_file.raw.class}"
puts
puts "io_set_text_file.iod.size: #{io_set_text_file.iod.size}"
puts "io_set_text_file.iod.class: #{io_set_text_file.iod.class}"
puts "io_set_text_file.iod.first.size: #{io_set_text_file.iod.first.size}"
puts "io_set_text_file.iod.first.class: #{io_set_text_file.iod.first.class}"
puts "io_set_text_file.iod.first.first.class: #{io_set_text_file.iod.first.first.class}"
# puts "io_set_text_file.iod.first.first.size: #{io_set_text_file.iod.first.first.size}"

puts
puts ":"*40
puts
# puts io_set_text_file.class.convert_iod_to_raw(io_set_text_file.iod)
# puts "inputs_sequence[0..49]: #{inputs_sequence[0..49].map{|v| v.map{|w| w.map{|x| x.round(8)}}}}"
# puts "outputs_sequence[0..49]: #{outputs_sequence[0..49].map{|v| v.map{|w| w.map{|x| x.round(8)}}}}"


puts "-"*40
puts

r = Runner.new(file_path)

r.compare_successive_training_rounds(
  io_offset, time_col_qty,
  inputs_sequence, outputs_sequence,
  hidden_layer_qty, hidden_size_given,
  qty_new_members,
  my_breed_manager, max_members,
  train_qty,
  io_set_text_file
)

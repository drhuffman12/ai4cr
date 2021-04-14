# Run via: `time CRYSTAL_WORKERS=24 crystal run examples/rnn_simple_manager_example.cr -Dpreview_mt --release > tmp/log.txt`
#   (Adjust the 'CRYSTAL_WORKERS=24' as desired.)
# Follow `tmp/log.txt' in your IDE or in console (i.e.: `tail -f tmp/log.txt`)
# Be on the look out for high `percent_correct: x of x` in the 'tmp/log.txt file'
# Monitor your Ram and CPU usage!
#   (This seems to stablize at around about 4 Gb and 1/3 of my system's AMD Ryzen 7 1700X CPU.)
# NOTE: Training results look promising, but tend to be more successful towards the 'more future' side of the outputs.
#   So, implement bi-directional RNN in the next phase, in hopes of balancing out the successfulness of the
#   'less future' vs 'more future' guesses.

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
    puts
    puts "v"*40
    puts "successive generations (should) score better (?) .. max_members: #{max_members} .. start"
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
      hidden_size_given: hidden_size_given,
      learning_styles: [Ai4cr::NeuralNetwork::LS_RELU]
    ).config

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
    p "."
    first_gen_members_scored = first_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
    first_gen_members_stats = first_gen_members.map { |member| member.error_hist_stats }

    p "."
    second_gen_members_scored = second_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
    second_gen_members_stats = second_gen_members.map { |member| member.error_hist_stats }

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
  end
end

####

my_breed_manager = Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager.new

file_path = "./spec_bench/support/neural_network/data/bible_utf/eng-web_002_GEN_01_read.txt"
file_type_raw = Ai4cr::Utils::IoData::FileType::Raw
prefix_raw_qty = 0
prefix_raw_char = " "
default_to_bit_size = 8

io_set_text_file = Ai4cr::Utils::IoData::TextFileIodBits.new(
  file_path, file_type_raw,
  prefix_raw_qty, prefix_raw_char,
  default_to_bit_size
)

# re 'compare_successive_training_rounds'
time_col_qty = 16      # 12 # 10 # 6 # 25
hidden_layer_qty = 3   # 4 # 6 # 3
hidden_size_given = 8 # 16 # 100 # 200
max_members = 10       # 5 # 10
train_qty = 3          # 1 # 2

io_offset = time_col_qty
ios = io_set_text_file.iod_to_io_set_with_offset_time_cols(time_col_qty, io_offset)

inputs_sequence = ios[:input_set]
outputs_sequence = ios[:output_set]
qty_new_members = max_members

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

r.compare_successive_training_rounds(
  io_offset, time_col_qty,
  inputs_sequence, outputs_sequence,
  hidden_layer_qty, hidden_size_given,
  qty_new_members,
  my_breed_manager, max_members,
  train_qty,
  io_set_text_file
)

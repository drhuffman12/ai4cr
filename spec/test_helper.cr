#
# test_helper.cr
#
# This is a unit test helper file for ai4cr
#
# Ported By:: Daniel Huffman
# Url::       https://github.com/drhuffman12/ai4cr
#
# Based on::  Ai4r
#   Author::    Olav Stetter
#   License::   MPL 1.1
#   Project::   ai4r
#   Url::       http://www.ai4r.org/
#
# You can redistribute it and/or modify it under the terms of
# the Mozilla Public License version 1.1  as published by the
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt
#

# TODO: add JSON-friendly versions of below!

def assert_approximate_equality(expected, real, delta = 0.01)
  if expected.responds_to?(:abs) && real.responds_to?(:abs)
    real.should be_close(expected, delta)
  else
    real.should eq(expected)
  end
end

def assert_approximate_inequality(expected, real, delta = 0.01)
  if expected.responds_to?(:abs) && real.responds_to?(:abs)
    real.should_not be_close(expected, delta)
  else
    real.should_not eq(expected)
  end
end

def assert_approximate_equality_of_nested_list(expected, real, delta = 0.01)
  if expected.responds_to?(:each) && real.responds_to?(:each) && expected.size == real.size
    [expected, real].transpose.each { |exer| ex = exer[0]; er = exer[1]; assert_approximate_equality_of_nested_list(ex, er, delta) }
  else
    assert_approximate_equality expected, real, delta
  end
end

def assert_equality_of_nested_list(expected, real)
  if expected.responds_to?(:each) && real.responds_to?(:each) && expected.size == real.size
    [expected, real].transpose.each { |exer| ex = exer[0]; er = exer[1]; assert_equality_of_nested_list(ex, er) }
  else
    real.should eq(expected)
  end
end

def assert_inequality_of_nested_list(expected, real)
  if expected.responds_to?(:each) && real.responds_to?(:each) && expected.size == real.size
    [expected, real].transpose.each { |exer| ex = exer[0]; er = exer[1]; assert_inequality_of_nested_list(ex, er) }
  else
    real.should_not eq(expected)
  end
end

def assert_approximate_inequality_of_nested_list(expected, real, delta = 0.01)
  if expected.responds_to?(:each) && real.responds_to?(:each) && expected.size == real.size
    [expected, real].transpose.each { |exer| ex = exer[0]; er = exer[1]; assert_approximate_inequality_of_nested_list(ex, er, delta) }
  else
    assert_approximate_inequality expected, real, delta
  end
end

# Author:: Daniel Huffman
# Url::    https://github.com/drhuffman12/ai4cr

def guess(net, raw_in)
  result = net.eval(raw_in)
  result.map(&.round(6))
end

def result_label(result)
  if result[0] > result[1] && result[0] > result[2]
    "TRIANGLE"
  elsif result[1] > result[2]
    "SQUARE"
  else
    "CROSS"
  end
end

def check_guess(next_guess, expected_label)
  result_label(next_guess).should eq(expected_label)
end

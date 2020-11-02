module Enumerable
  # def sum
  #   self.reduce(0.0){|acc, i| acc + i }
  # end

  def mean
    1.0 * self.sum / self.size
  end

  def sample_variance
    m = self.mean
    psum = self.reduce(0.0) { |acc, i| acc + (i - m)**2.0 }
    psum/(self.size - 1).to_f
  end

  def standard_deviation
    Math.sqrt(self.sample_variance)
  end
end
/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define NUM_OF_PARTICLES 1000
#define SIGMA .0001
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = NUM_OF_PARTICLES;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i =0; i < num_particles; i++)
	{
		Particle P;
		P.x = dist_x(gen);
		P.y = dist_y(gen);
		P.theta = dist_theta(gen);
		P.weight = 1.0;
		P.id = i;

		particles.push_back(P);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	double predicted_x, predicted_y, predicted_theta;

	for (int i = 0; i < num_particles; i++)
	{
		if (abs(yaw_rate) < SIGMA)
		{
			predicted_x = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
			predicted_y = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
			predicted_theta = particles[i].theta;
		}
		else
		{
			predicted_x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
			predicted_y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			predicted_theta = particles[i].theta + (yaw_rate * delta_t);
		}

		default_random_engine gen;
		normal_distribution<double> dist_x(predicted_x, std_pos[0]);
		normal_distribution<double> dist_y(predicted_y, std_pos[1]);
		normal_distribution<double> dist_theta(predicted_theta, std_pos[2]);

		particles[i].x = dist_x(gen);
	    particles[i].y = dist_y(gen);
	    particles[i].theta = dist_theta(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i = 0; i < observations.size(); i++)
	{
		int closest_obs_id = -1;
		double lowest_dist = 100000;
		for (unsigned int j = 0; j < predicted.size(); j++)
		{
			double distance = dist(observations[i].x,observations[i].y,
								predicted[j].x, predicted[j].y);

			if (distance < lowest_dist)
			{
				lowest_dist = distance;
				closest_obs_id = predicted[j].id;
			}
		}
		observations[i].id = closest_obs_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double weight_normalizer = 0.0;
	for (int i = 0; i < num_particles; i++)
	{

		/* Transforming observations from particle coordinates to map coordinates */
		/* Xm = Xp +(cosθ×Xc)−(sinθ×Yc) */
		/* Ym = Yp +(sinθ×Xc)+(cosθ×Yc) */
		std::vector<LandmarkObs> transformed_obs;
		for(int j = 0; j < observations.size(); j++)
		{
			LandmarkObs obs;
		    obs.id = j;
		    obs.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
		    obs.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
		    transformed_obs.push_back(obs);

		}

		/*Create new map with the observed landmarks by the sensor range only*/
		std::vector<LandmarkObs> filtered_map;
		double distance;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			distance = dist(particles[i].x, particles[i].y,
							map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			if(distance < sensor_range)
			{
				LandmarkObs obs;
				obs.id = map_landmarks.landmark_list[j].id_i;
				obs.x = map_landmarks.landmark_list[j].x_f;
				obs.y = map_landmarks.landmark_list[j].y_f;
				filtered_map.push_back(obs);
			}
		}

		/*Assosiate observations with map land marks*/
		dataAssociation(filtered_map, transformed_obs);

		/*Step 4: Calculate the weight of each particle using Multivariate Gaussian distribution.*/
		//Reset the weight of particle to 1.0
		particles[i].weight = 1.0;


		double sigma_x = std_landmark[0];
		double sigma_y = std_landmark[1];
		double sigma_x_2 = pow(sigma_x, 2);
		double sigma_y_2 = pow(sigma_y, 2);
		double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
		int k, l;

		/*Calculate the weight of particle based on the multivariate Gaussian probability function*/
		for (k = 0; k < transformed_obs.size(); k++) {
		  double trans_obs_x = transformed_obs[k].x;
		  double trans_obs_y = transformed_obs[k].y;
		  double trans_obs_id = transformed_obs[k].id;
		  double multi_prob = 1.0;

		  for (l = 0; l < filtered_map.size(); l++) {
			double pred_landmark_x = filtered_map[l].x;
			double pred_landmark_y = filtered_map[l].y;
			double pred_landmark_id = filtered_map[l].id;

			if (trans_obs_id == pred_landmark_id) {
			  multi_prob = normalizer * exp(-1.0 * ((pow((trans_obs_x - pred_landmark_x), 2)/(2.0 * sigma_x_2)) + (pow((trans_obs_y - pred_landmark_y), 2)/(2.0 * sigma_y_2))));
			  particles[i].weight *= multi_prob;
			}
		  }
		}
		weight_normalizer += particles[i].weight;
	  }

	  /*Step 5: Normalize the weights of all particles since resmapling using probabilistic approach.*/
	  for (int i = 0; i < particles.size(); i++) {
		particles[i].weight /= weight_normalizer;
		weights.push_back(particles[i].weight);
	  }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resampled_particles;

	// Create a generator to be used for generating random particle index and beta value
	default_random_engine gen;

	//Generate random particle index
	uniform_int_distribution<int> particle_index(0, num_particles - 1);

	int current_index = particle_index(gen);

	double beta = 0.0;

	double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());

	for (int i = 0; i < particles.size(); i++) {
		uniform_real_distribution<double> random_weight(0.0, max_weight_2);
		beta += random_weight(gen);

	  while (beta > weights[current_index]) {
		beta -= weights[current_index];
		current_index = (current_index + 1) % num_particles;
	  }
	  resampled_particles.push_back(particles[current_index]);
	}
	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

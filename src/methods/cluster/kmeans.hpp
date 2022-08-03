#ifndef METHODS_CLUSTER_KMEANS_HPP
#define METHODS_CLUSTER_KMEANS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace cluster {

/**
 * K-Means clustering.
*/
template<typename DataType>
class KMeans {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    
    std::string init_;
    std::size_t num_init_;
    std::size_t num_clusters_;
    std::size_t max_iter_;
    double tol_;
    
    MatType centroids_;

    /**
     * kmeans++ initialization of clusters
    */
    const MatType kmeans_plusplus(const MatType& X) const {
        std::size_t num_samples = X.rows(), num_features = X.cols();
        // copy from sklean tried, no specific results 
        // for other than mentioning in the conclusion
        std::size_t num_local_trials = 2 + 
            static_cast<std::size_t>(std::log(num_clusters_));
        
        // generate randomly first index of center
        auto center_index = random::randint<std::size_t>(1, 1, 0, num_samples - 1);

        MatType centers(num_clusters_, num_features);
        centers.row(0) = X.row(center_index.value());

        // compute the distance between all sample point and the first center point
        MatType diff1 = X - utils::repeat<MatType>(centers.row(0), num_samples, 0);
        VecType closest_dist = math::row_norms<MatType, VecType>(diff1, true);

        // compute the sume of distance, this param allows map a random value of 
        // domaine interval [0, 1] to a random intervall [0, current_pot]
        auto tmp = math::sum<MatType, VecType>(closest_dist, -1);
        DataType current_pot = tmp.value();
        for (std::size_t c = 1; c < num_clusters_; ++c) {
            // Choose center candidates by sampling
            VecType rand_vec = random::rand<MatType>(num_local_trials, 1);
            DataType candidates_pot = ConstType<DataType>::max();
            std::size_t best_candidates;
            VecType closest_dist_to_candidates;

            for (std::size_t i = 0; i < num_local_trials; ++i) {
                
                DataType rand_val = rand_vec(i, 0) * current_pot;
                // cumulative the closest distances
                VecType cum_closest_dist = math::cumsum<MatType, VecType>(closest_dist, -1);

                // find the first index of the related value what is more than current random value
                auto lower = std::lower_bound(
                    cum_closest_dist.begin(), cum_closest_dist.end(), rand_val
                );
                std::size_t candidate_index = static_cast<std::size_t>(
                    std::distance(cum_closest_dist.begin(), lower)
                );
                // numerical imprecision can result in a candidate_id out of range
                if (candidate_index > closest_dist.size() - 1) {
                    candidate_index = closest_dist.size() - 1;
                }
                //  Compute distances to center candidates
                MatType diff2 = X - utils::repeat<MatType>(X.row(candidate_index), num_samples, 0);
                VecType dist_to_candidates = math::row_norms<MatType, VecType>(diff2, true);

                // update closest distances squared and potential for each candidate
                for (int i = 0; i < num_samples; ++i) {
                    if (closest_dist(i, 0) < dist_to_candidates(i, 0)) {
                        dist_to_candidates(i, 0) = closest_dist(i, 0);
                    }
                }
                // Calculate the range values for all candidate prime roulette selection mappings 
                // since minimum distance from each sample to the prime was updated in the previous step
                DataType current_candidates_pot = dist_to_candidates.array().sum();
                // choose which candidate is the best
                if (current_candidates_pot < candidates_pot) {
                    candidates_pot = current_candidates_pot;
                    best_candidates = candidate_index;
                    closest_dist_to_candidates = dist_to_candidates;
                }
            }
            current_pot = candidates_pot;
            closest_dist = closest_dist_to_candidates;
            centers.row(c) = X.row(best_candidates);
        }
        return centers;
    }



protected:
    void init_centroid(const MatType& X, 
        const VecType& x_squared_norms) {
        
        std::size_t num_samples = X.rows(), num_features = X.cols(); 
        MatType centroids(num_clusters_, num_features);

        if (init_ == "random") {
            IdxType index = random::permutation<IdxType>(num_samples);
            IdxType selected_index = index.topRows(num_clusters_);
            centroids = X(selected_index, Eigen::all);
        } 
        else if (init_ == "kmeans++") {
            centroids = kmeans_plusplus(X);
        }

        centroids_ = centroids;
    }
    

    /**
     * k-means lloyd
    */
    void kmeans_lloyd(const MatType& X) {

        // bool converged = false;
        std::size_t num_samples = X.rows(), num_features = X.cols(); 
        // MatType new_centroid(num_clusters_, num_features);

        for (std::size_t iter = 0; iter < max_iter_; ++iter) {
            // define the cluster
            std::map<std::size_t, std::vector<std::size_t>> clusters;
            for (std::size_t i = 0; i < num_samples; ++i) {
                std::size_t min_dist_index = 0;
                double min_dist = ConstType<double>::infinity();
                // compute the min distance between sample and centroid
                for (std::size_t j = 0; j < num_clusters_; ++j) {
                    MatType tmp;
                    tmp = X.row(i).array() - centroids_.row(j).array();
                    double dist = static_cast<double>(tmp.norm());
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_dist_index = j;
                    }
                }
                clusters[min_dist_index].push_back(i);
            }
            // move the centers 
            double eps = 0.0; 
            for (auto& c : clusters) {
                MatType cluster = X(c.second, Eigen::all);
                MatType mean_c;
                mean_c = math::mean<MatType, VecType>(cluster, 0);

                MatType diff;
                diff = mean_c.array() - centroids_.row(c.first).array();
                
                eps += static_cast<double>(diff.norm());
                centroids_.row(c.first) = mean_c.transpose();
            }
            // convergen condition
            if (eps < tol_) {
                break;
            }
        }
    }

    /**
     * predict label
    */
    const VecType predict_label(const MatType& X) const{
        std::size_t num_samples = X.rows();
        VecType y_pred(num_samples);
        for (std::size_t i = 0; i < num_samples; ++i) {
            std::size_t min_dist_index = 0;
            double min_dist = ConstType<double>::infinity();

            for (std::size_t j = 0; j < num_clusters_; ++j) {
                MatType diff;
                diff = X.row(i).array() - centroids_.row(j).array();
                double dist = static_cast<double>(diff.norm());
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_index = j;
                }
            }
            y_pred(i) = min_dist_index;
        }
        return y_pred;
    }




public:
    KMeans(): init_("random"), 
        num_init_(10),
        num_clusters_(3), 
        max_iter_(300), 
        tol_(1e-4) {};

    KMeans(std::string init,
        std::size_t num_init,
        std::size_t num_clusters,
        std::size_t max_iter, 
        double tol): init_(init), 
            num_init_(num_init),
            num_clusters_(num_clusters), 
            max_iter_(max_iter), 
            tol_(tol) {};


    void test_func(const MatType& X) {
        MatType centroids;

        init_centroid(X, VecType());

        // std::cout << "centroids" << std::endl;
        // std::cout << centroids << std::endl;


        // MatType new_centroids;
        kmeans_lloyd(X);

        VecType y_pred = predict_label(X);

        std::cout << "y_pred" << std::endl;
        std::cout << y_pred << std::endl;
    }





};

} // cluster_model
} // openml

#endif /*METHODS_CLUSTER_KMEANS_HPP*/

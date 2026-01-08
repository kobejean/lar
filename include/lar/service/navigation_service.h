#ifndef LAR_SERVICE_NAVIGATION_SERVICE_H
#define LAR_SERVICE_NAVIGATION_SERVICE_H

#include <grpcpp/grpcpp.h>
#include "navigation.grpc.pb.h"
#include "lar/core/map.h"

namespace lar {

class NavigationServiceImpl final : public proto::NavigationService::Service {
public:
    explicit NavigationServiceImpl(Map& map);

    grpc::Status GetPath(grpc::ServerContext* context,
                         const proto::GetPathRequest* request,
                         proto::GetPathResponse* response) override;

    grpc::Status GetLandmarks(grpc::ServerContext* context,
                              const proto::GetLandmarksRequest* request,
                              proto::GetLandmarksResponse* response) override;

private:
    Map& map_;
};

} // namespace lar

#endif // LAR_SERVICE_NAVIGATION_SERVICE_H
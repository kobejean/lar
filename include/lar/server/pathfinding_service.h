#ifndef LAR_SERVER_PATHFINDING_SERVICE_H
#define LAR_SERVER_PATHFINDING_SERVICE_H

#include <grpcpp/grpcpp.h>
#include "pathfinding.grpc.pb.h"
#include "lar/core/map.h"

namespace lar {

class PathfindingServiceImpl final : public proto::PathfindingService::Service {
public:
    explicit PathfindingServiceImpl(Map& map);

    grpc::Status GetPath(grpc::ServerContext* context,
                         const proto::GetPathRequest* request,
                         proto::GetPathResponse* response) override;

private:
    Map& map_;
};

} // namespace lar

#endif // LAR_SERVER_PATHFINDING_SERVICE_H

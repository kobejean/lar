#include "lar/server/pathfinding_service.h"

namespace lar {

PathfindingServiceImpl::PathfindingServiceImpl(Map& map) : map_(map) {}

grpc::Status PathfindingServiceImpl::GetPath(
    grpc::ServerContext* /*context*/,
    const proto::GetPathRequest* request,
    proto::GetPathResponse* response) {

    auto path = map_.getPath(request->start_id(), request->goal_id());

    response->set_found(!path.empty());

    for (const Anchor* anchor : path) {
        auto* proto_anchor = response->add_path();
        proto_anchor->set_id(anchor->id);
        proto_anchor->set_x(anchor->transform.translation().x());
        proto_anchor->set_y(anchor->transform.translation().y());
        proto_anchor->set_z(anchor->transform.translation().z());
    }

    return grpc::Status::OK;
}

} // namespace lar
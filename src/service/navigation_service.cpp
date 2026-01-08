#include "lar/service/navigation_service.h"
#include "lar/core/spatial/rect.h"

namespace lar {

NavigationServiceImpl::NavigationServiceImpl(Map& map) : map_(map) {}

grpc::Status NavigationServiceImpl::GetPath(
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

grpc::Status NavigationServiceImpl::GetLandmarks(
    grpc::ServerContext* /*context*/,
    const proto::GetLandmarksRequest* request,
    proto::GetLandmarksResponse* response) {

    // Convert proto Rect to lar::Rect
    Rect query(
        request->query().lower().x(), request->query().lower().y(),
        request->query().upper().x(), request->query().upper().y()
    );

    // Find landmarks in the query region
    std::vector<Landmark*> results;
    map_.landmarks.find(query, results, request->limit());

    // Convert results to proto Landmark messages
    for (const Landmark* landmark : results) {
        auto* proto_landmark = response->add_landmarks();
        proto_landmark->set_id(landmark->id);
        proto_landmark->set_x(landmark->position.x());
        proto_landmark->set_y(landmark->position.y());
        proto_landmark->set_z(landmark->position.z());
        proto_landmark->set_desc(
            landmark->desc.data,
            landmark->desc.total() * landmark->desc.elemSize()
        );
    }

    return grpc::Status::OK;
}

} // namespace lar
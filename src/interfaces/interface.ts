export interface Trip {
  TripID: string;
  UserID: string;
  PickupAddress: string;
  DropoffAddress: string;
  PickupTime: string;
  DropoffTime: string;
  NumRiders: number;
  riders?: string[];
  ages?: number[];
}

export interface Rider {
  TripID: string;
  UserID: string;
}

export interface UserAge {
  UserID: string;
  Age: number;
}
